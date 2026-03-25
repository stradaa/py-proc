import re
import io
import sys
import time
import zlib
import json
import shutil
import typing
import struct
import pickle
import pathlib
import argparse
import threading
import traceback
import itertools
import contextlib
import subprocess
import collections
from multiprocessing.pool import ThreadPool

import yaml
import numpy
import scipy.io

from thalamus.thalamus_pb2 import StorageRecord, Image, Compressed
import google.protobuf.message

# python procThalamus_indie.py -d 260324 --skip-video

EXECUTABLE_EXTENSION = '.exe' if sys.platform == 'win32' else ''

LONG = ">Q"
LONG_SIZE = struct.calcsize(LONG)
MAX_SIZE = 40e6

class PendingMessage(typing.NamedTuple):
  size: int
  type: int
  stream: int

class ZQueue:
  def __init__(self, stream: int):
    self.inflater = zlib.decompressobj()
    self.messages: collections.deque[Compressed] = collections.deque()
    self.pending_messages: collections.deque[PendingMessage] = collections.deque()
    self.lock = threading.Lock()
    self.done = False
    self.working = False
    self.stream_id = stream
    self.output_buffer = b''
    self.output_messages: collections.deque[StorageRecord] = collections.deque()
    self.gg = 0

  def push(self, message: Compressed):
    with self.lock:
      self.messages.append(message)

  def pull(self):
    with self.lock:
      while not self.output_messages and not self.done:
        self.lock.release()
        time.sleep(1)
        self.lock.acquire()
      if self.done:
        return None
      return self.output_messages.popleft()

  def work(self):
    try:
      while True:
        buffer = b''
        with self.lock:
          if self.working or self.done:
            return
          messages = self.messages
          self.messages = collections.deque()
          self.working = bool(messages)

        for m in messages:
          buffer += m.data
          if m.type == Compressed.Type.NONE:
            continue
          self.pending_messages.append(PendingMessage(m.size, m.type, m.stream))

        if not buffer:
          with self.lock:
            self.working = False
            return

        new_output = self.inflater.decompress(buffer)
        self.output_buffer += new_output
        new_output_messages = []
        while self.pending_messages:
          pending = self.pending_messages[0]
          if pending.size > len(self.output_buffer):
            break

          self.pending_messages.popleft()

          try:
            message = StorageRecord()
            message.ParseFromString(self.output_buffer[:pending.size])
            new_output_messages.append(message)
            self.output_buffer = self.output_buffer[pending.size:]
            self.gg += 1
          except google.protobuf.message.DecodeError:
            traceback.print_exc()
            with self.lock:
              self.done = True
            break
        with self.lock:
          self.output_messages.extend(new_output_messages)
          self.working = False
    except:
      traceback.print_exc()

class RecordReader:
  def __init__(self, file_arg: typing.Union[str, pathlib.Path, io.BufferedReader], decompress=True, mux=True):
    self.filename: typing.Optional[pathlib.Path]
    self.reader: typing.Optional[io.BufferedReader]
    self.size = 0
    self.current_position = 0
    self.decompress = decompress
    self.mux = mux
    self.muxers: typing.Dict[str, subprocess.Popen] = {}
    self.running = False
    self.pool = ThreadPool()
    self.thread: typing.Optional[threading.Thread] = None
    self.condition = threading.Condition()
    self.lock = threading.Lock()
    self.z_queues = {}
    self.image_nodes: typing.List[str] = []
    self.records: collections.deque[typing.Tuple[int, typing.Union[StorageRecord, PendingMessage, None]]] = collections.deque()
    if isinstance(file_arg, (str, pathlib.Path)):
      self.filename = pathlib.Path(file_arg)
      self.reader = None
    else:
      self.filename = None
      self.reader = file_arg
      self.measure()

  def is_running(self):
    with self.lock:
      return self.running

  def reader_thread(self):
    muxers: typing.Dict[str, subprocess.Popen] = {}
    assert self.reader is not None
    try:
      while self.is_running():
        record = self.__read_record()
        position = self.reader.tell()
        if record is None:
          with self.lock:
            self.records.append((self.size, None))
          return

        body_type = record.WhichOneof('body')
        if body_type == 'compressed':
          if not self.decompress:
            with self.lock:
              self.records.append((position, record))
              continue

          compressed = record.compressed
          with self.lock:
            if compressed.stream not in self.z_queues:
              self.z_queues[compressed.stream] = ZQueue(compressed.stream)
            z_queue = self.z_queues[compressed.stream]
            z_queue.push(compressed)
            self.pool.apply_async(z_queue.work)

          if compressed.type == Compressed.Type.NONE:
            continue

          with self.lock:
            self.records.append((position, PendingMessage(compressed.size, compressed.type, compressed.stream)))
        elif body_type == 'image':
          image = record.image
          if image.format in (Image.Format.MPEG1, Image.Format.MPEG4, Image.Format.Gray, Image.Format.RGB) and self.mux:
            if record.node not in muxers:
              self.image_nodes.append(record.node)
              output_file = f'{self.filename}.{record.node}.avi'
              if image.format in (Image.Format.MPEG1, Image.Format.MPEG4,):
                muxers[record.node] = subprocess.Popen(f'ffmpeg -y -i pipe: -c:v copy "{output_file}"', stdin=subprocess.PIPE, shell=True)
              elif image.format in (Image.Format.Gray, Image.Format.RGB):
                framerates = [
                  (24000.0 / 1001, "24000/1001"),
                  (24, "24"),
                  (25, "25"),
                  (30000.0 / 1001, "30000/1001"),
                  (30, "30"),
                  (50, "50"),
                  (60000.0 / 1001, "60000/1001"),
                  (60, "60"),
                  (15, "15"),
                  (5, "5"),
                  (10, "10"),
                  (12, "12"),
                  (15, "15")
                ]
                framerate = min(framerates, key=lambda a: abs(a[0] - 1e9/(image.frame_interval or 16e6)))
                if image.format == Image.Format.Gray:
                  format = 'gray'
                else:
                  format = 'rgb24'
                command = (f'ffmpeg -y -f rawvideo -r {framerate[1]} -pixel_format {format} -video_size {image.width}x{image.height} '
                           f'-i pipe: -qscale:v 2 -b:v 100M "{output_file}"')
                print('COMMAND', command)
                muxers[record.node] = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)
            muxer = muxers[record.node]
            assert muxer.stdin is not None
            if len(image.data) > 0:
              muxer.stdin.write(image.data[0])
            if image.width > 0:
              with self.lock:
                self.records.append((position, record))
          else:
            with self.lock:
              self.records.append((position, record))
        else:
          with self.lock:
            self.records.append((position, record))
    except:
      traceback.print_exc()
    finally:
      for k, v in muxers.items():
        assert v.stdin is not None
        v.stdin.close()
      for k, v in muxers.items():
        v.wait()

  def get_record(self) -> typing.Optional[StorageRecord]:
    with self.lock:
      while not self.records:
        self.lock.release()
        time.sleep(1)
        self.lock.acquire()
      position, record = self.records.popleft()
      if isinstance(record, PendingMessage):
        z_queue = self.z_queues[record.stream]
        self.lock.release()
        record = z_queue.pull()
        self.lock.acquire()
      self.current_position = position
      return record

  def progress(self):
    return self.current_position/self.size

  def read_progress(self):
    assert self.reader is not None
    return self.reader.tell()/self.size

  def start(self):
    with self.lock:
      self.running = True
    self.pool.__enter__()
    self.pool.apply_async(self.reader_thread)
    print('start')

  def stop(self, type = None, value = None, tb = None):
    with self.lock:
      self.running = False
    self.pool.__exit__(type, value, tb)

  def __enter__(self):
    if self.reader is None:
      assert self.filename is not None
      self.reader = open(self.filename, 'rb')
      self.filename = pathlib.Path(self.reader.name)
      self.measure()
    self.start()
    return self

  def __exit__(self, type, value, tb):
    self.stop(type, value, tb)

  def measure(self):
    assert self.reader is not None
    self.reader.seek(0, 2)
    self.size = self.reader.tell()
    self.reader.seek(0, 0)

  def __read_record(self) -> typing.Optional[StorageRecord]:
    assert self.reader is not None
    data = self.reader.read(LONG_SIZE)
    if not data:
      return

    size, = struct.unpack(LONG, data)
    if size > MAX_SIZE:
      return

    data = self.reader.read(size)
    message = StorageRecord()

    try:
      message.ParseFromString(data)
      return message
    except google.protobuf.message.DecodeError:
      return

  def __iter__(self) -> 'RecordReader':
    return self

  def __next__(self) -> StorageRecord:
    record = self.get_record()
    if record is None:
      raise StopIteration()
    return record

class Timer:
  def __init__(self, seconds: float):
    self.seconds = seconds
    self.callbacks: typing.List[typing.Callable[[], None]] = []
    self.last_time = 0.0
    self.reset()

  def reset(self):
    self.last_time = time.perf_counter()

  def add_callback(self, callback: typing.Callable[[], None]):
    self.callbacks.append(callback)

  def poll(self):
    now = time.perf_counter()
    if now - self.last_time >= self.seconds:
      for c in self.callbacks:
        c()
      self.last_time = now

def read_record(stream) -> typing.Optional[StorageRecord]:
  data = stream.read(LONG_SIZE)
  if not data:
    return

  size, = struct.unpack(LONG, data)
  if size > MAX_SIZE:
    return

  data = stream.read(size)
  message = StorageRecord()

  try:
    message.ParseFromString(data)
    return message
  except google.protobuf.message.DecodeError:
    return

def is_capturefile(f: pathlib.Path):
  if not f.is_file():
    return False
  with open(f, 'rb') as stream:
    return read_record(stream) is not None

class Reward(typing.NamedTuple):
  timestamp_ns: int
  on_time_ns: int

class OculomaticPoint(typing.NamedTuple):
  x: float = sys.maxsize
  y: float = sys.maxsize
  diameter: float = sys.maxsize
  width: int = sys.maxsize
  height: int = sys.maxsize

class HandPoint(typing.NamedTuple):
  x: float = sys.maxsize
  y: float = sys.maxsize

class JoystickSample(typing.NamedTuple):
  timestamp_ns: int
  x: float
  y: float

def _extract_joystick_samples(record_time_ns: int, analog) -> typing.List[JoystickSample]:
  """
  Extract ordered joystick samples from a Thalamus analog record.

  Preserves every sample instead of collapsing by record timestamp. When a
  record contains multiple X/Y samples, timestamps are reconstructed from
  sample_intervals if possible; otherwise the record timestamp is used for all
  samples in that packet.
  """
  x_vals = None
  y_vals = None
  for span in analog.spans:
    if span.name == 'X':
      x_vals = numpy.asarray(analog.data[span.begin:span.end], dtype=float)
    elif span.name == 'Y':
      y_vals = numpy.asarray(analog.data[span.begin:span.end], dtype=float)

  if x_vals is None and y_vals is None:
    return []

  if x_vals is None:
    x_vals = numpy.full(len(y_vals), numpy.nan, dtype=float)
  if y_vals is None:
    y_vals = numpy.full(len(x_vals), numpy.nan, dtype=float)

  n_samples = max(len(x_vals), len(y_vals))
  if n_samples == 0:
    return []

  def _normalize_channel(arr: numpy.ndarray) -> numpy.ndarray:
    if len(arr) == n_samples:
      return arr
    if len(arr) == 1:
      return numpy.repeat(arr, n_samples)
    out = numpy.full(n_samples, numpy.nan, dtype=float)
    out[:min(len(arr), n_samples)] = arr[:min(len(arr), n_samples)]
    return out

  x_vals = _normalize_channel(x_vals)
  y_vals = _normalize_channel(y_vals)

  timestamps = numpy.full(n_samples, int(record_time_ns), dtype=numpy.int64)
  sample_intervals = numpy.asarray(analog.sample_intervals, dtype=numpy.int64)
  if n_samples > 1 and len(sample_intervals) > 0:
    if len(sample_intervals) == 1:
      offsets = numpy.arange(n_samples, dtype=numpy.int64) * sample_intervals[0]
      timestamps = int(record_time_ns) + offsets
    else:
      usable = sample_intervals[:max(0, n_samples - 1)]
      if len(usable) > 0:
        offsets = numpy.concatenate([[0], numpy.cumsum(usable, dtype=numpy.int64)])
        timestamps[:len(offsets)] = int(record_time_ns) + offsets[:n_samples]
        if len(offsets) < n_samples:
          timestamps[len(offsets):] = timestamps[len(offsets) - 1]

  return [
    JoystickSample(int(ts), float(x), float(y))
    for ts, x, y in zip(timestamps, x_vals, y_vals)
  ]

def get_rec_number(path: pathlib.Path):
  if path.suffix == '.novideo':
    return int(path.with_suffix('').suffix[1:])
  else:
    return int(path.suffix[1:])

def main():
  parser = argparse.ArgumentParser(
                    prog='Thalamus file hydrater (AlexRig)',
                    description='Transforms thalamus files into HDF5 - adapted for AlexRig (single-file recording)')
  parser.add_argument('-d', '--day-dir', type=pathlib.Path)
  parser.add_argument('-s', '--skip-video', action='store_true')
  parser.add_argument('--skip-existing-video', action='store_true',
                      help='Skip camera nodes whose MP4 and mat outputs already exist')
  args = parser.parse_args()
  day_dir = typing.cast(pathlib.Path, args.day_dir)

  print(args)

  recs: typing.List[pathlib.Path] = []
  behave_configs: typing.Dict[str, typing.Any] = {}
  bad_recs = []
  for f in day_dir.iterdir():
    print(f)
    if f.name.startswith('behave') and f.suffix == '.json':
      with open(f) as config_file:
        try:
          behave_configs[f.with_suffix('').name + ('.novideo' if args.skip_video else '')] = json.load(config_file)
        except json.JSONDecodeError:
          bad_recs.append(f.with_suffix(''))
    elif f.name.startswith('behave') and is_capturefile(f) and ((f.suffix == '.novideo') if args.skip_video else (f.suffix != '.novideo')):
      recs.append(f)
  recs = set(recs) - set(bad_recs)
  recs = sorted(recs, key=lambda r: r.name)
  print('recs:', recs)
  print('configs:', list(behave_configs.keys()))

  # Get NIDAQ sample rate from behave config (no separate recorder.json)
  nidaq_sample_rate = None
  for config in behave_configs.values():
    for node in config['nodes']:
      if node['type'] == 'NIDAQ':
        nidaq_sample_rate = node.get('Sample Rate')
        break
    if nidaq_sample_rate is not None:
      break

  if nidaq_sample_rate is None:
    print('WARNING: Could not find NIDAQ Sample Rate in any behave config')
    nidaq_sample_rate = 1000.0

  print(f'NIDAQ Sample Rate: {nidaq_sample_rate}')

  ACQUISITION_TYPE = numpy.dtype([
    ('type', numpy.str_, 128),
    ('name', numpy.str_, 128),
    ('data_format', numpy.str_, 128),
    ('samplingrate', numpy.float64)
  ])

  acquisition = numpy.array(('ros2_recorder', '/recorder_node', 'single', nidaq_sample_rate), dtype=ACQUISITION_TYPE)

  HARDWARE_TYPE = numpy.dtype([
    ('type', numpy.str_, 128),
    ('acquisition', ACQUISITION_TYPE),
  ])

  EXPERIMENT_TYPE = numpy.dtype([
    ('hardware', HARDWARE_TYPE),
  ])

  REC_META_TYPE = numpy.dtype([
    ('Fs_rec', numpy.float64),
    ('data_format', numpy.str_, 16),
  ])

  hardware_def = numpy.array([('ros', acquisition)], dtype=HARDWARE_TYPE)
  experiment_def = numpy.array([(hardware_def,)], dtype=EXPERIMENT_TYPE)

  for rec in recs:
    rec_number = get_rec_number(rec)
    rec_number = f'{rec_number:03}'
    rec_dir = day_dir / rec_number
    rec_dir.mkdir(exist_ok=True, parents=True)

  # Experiment definition - look in monkeydir (e.g. Bowser_Behavior_AlexRig/Bowser_Behavior_AlexRig/)
  monkey_dir = day_dir.parent / day_dir.parent.name
  experiment_def_file = monkey_dir / 'prototype.experiment.mat'
  if not experiment_def_file.exists():
    experiment_def_file = None
    print(f'WARNING: prototype.experiment.mat not found in {monkey_dir}')
  print('experiment_def', experiment_def_file)

  etl_config_path = day_dir / "config.json"
  if etl_config_path.exists():
    with open(etl_config_path) as etl_config_file:
      etl_config = json.load(etl_config_file)
  else:
    etl_config = {}

  # AlexRig channel mapping for Analog in (NIDAQ)
  FIDUCIAL_CHANNEL = 'Dev1/ai0'
  DISPLAY_CHANNEL = 'Dev1/ai1'
  REWARD_DAQ_CHANNEL = 'Dev1/ai2'

  print('Measure time')
  rec_times: typing.List[typing.Tuple[int, int]] = []
  for rec in recs:
    behave_config = behave_configs[rec.name] if rec.name in behave_configs else list(behave_configs.values())[0]

    # Find touch transform — must be the TOUCH_SCREEN node whose Source is 'Node 2'
    # (the calibrated node driven by the Windows remote).  The vestigial 'Touch Screen'
    # node (Source='Touch Picker') may appear first in the list and must be skipped.
    touch_transform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # identity default
    for node in behave_config['nodes']:
      if node['type'] == 'TOUCH_SCREEN' and 'Transform' in node and node.get('Source') == 'Node 2':
        touch_transform = numpy.array(node['Transform']).T.flatten().tolist()
        break

    # Find wallclock name
    wallclock_name = None
    for node in behave_config['nodes']:
      if node['type'] == 'WALLCLOCK':
        wallclock_name = node['name']

    rec_number = get_rec_number(rec)
    rec_int = rec_number
    rec_number = f'{rec_number:03}'
    rec_dir = day_dir / rec_number
    print(rec)

    # --- Behavioral data ---
    image_times = collections.defaultdict(lambda: [])
    rewards: typing.List[Reward] = []
    oculomatic: typing.Dict[int, OculomaticPoint] = collections.defaultdict(OculomaticPoint)
    hand: typing.Dict[int, HandPoint] = collections.defaultdict(HandPoint)
    joystick: typing.List[JoystickSample] = []
    trial_summary_time = []
    behav_state_time = []
    behav_state = []
    task_config = []
    task_result = []
    used_values = []
    behave_result = []

    dims: typing.Dict[str, typing.List[typing.Tuple[int, int]]] = collections.defaultdict(lambda: [])
    message_count = 0
    success_count = 0
    fg = 1
    wall_time = 0

    # --- NIDAQ / recorder data (now in same file) ---
    fiducial = []        # Fiducial node timestamps (if node is present)
    fiducial_pulse = []  # Fiducial node pulse timestamps (if node is present)
    fiducial_daq = []    # Raw DAQ fiducial signal from Analog in
    sample_interval = sys.maxsize
    display_samples = []
    # Upsample factor for DAQ fiducial signal to match original procThalamus behavior.
    # This may not be necessary for AlexRig since there is no separate recorder sample rate,
    # but is kept for compatibility with downstream processing.
    upsample_factor = 30

    # --- Hand / touchscreen data ---
    # On AlexRig, scaled hand position comes from Node 3 (TOUCH_SCREEN with calibration transform).
    # Node 2 (SERIAL_TOUCH_SCREEN) provides raw serial coordinates; Node 3 transforms them.
    # Downstream proc assumes hnd.dat is RAW x, y, not transformed. So we select Node 2
    # Saving raw data enables posthoc recalibration if needed, etc.
    #

    HAND_NODE = 'Node 2'

    # Single pass over the behave file - reads both behavioral and NIDAQ data
    with RecordReader(rec, False, not args.skip_video) as record_reader:
      timer = Timer(1)
      timer.add_callback(lambda: print(100*record_reader.progress()))
      time_range = int(1e100), int(-1e100)
      for record in record_reader:
        message_count += 1
        timer.poll()
        body = record.WhichOneof('body')

        if body == 'text':
          text = record.text
          if text.text.startswith('BehavState='):
            print((record.time - time_range[0])/1e9, text.text, fg)
            fg += 1
            tokens = text.text.split('=')
            if tokens[1].lower() == 'success':
              success_count += 1
            behav_state.append(tokens[1])
            behav_state_time.append(text.time)
          else:
            try:
              doc = json.loads(text.text)
              if 'task_config' in doc:
                new_task_config = doc['task_config']
                print('====================', new_task_config['task_type'], success_count)
                new_task_config['touch_tform'] = touch_transform
                task_config.append(json.dumps(new_task_config))
                task_result.append(json.dumps(doc['task_result']))
                used_values.append(json.dumps(doc['used_values']))
                behave_result.append(json.dumps(doc.get('behav_result', '')))
                trial_summary_time.append(text.time)
            except ValueError:
              pass

        elif body == 'image':
          image = record.image
          image_times[record.node].append(record.time)
          dims[record.node].append((image.width, image.height))
          if record.node == 'Oculomatic':
            current = oculomatic[record.time]
            oculomatic[record.time] = current._replace(width=image.width, height=image.height)

        elif body == 'analog':
          analog = record.analog

          if wallclock_name and record.node == wallclock_name:
            if not wall_time:
              print('Wall Clock')
              wall_time = analog.data[0]

          elif record.node == 'Reward':
            rewards.append(Reward(record.time, analog.sample_intervals[0]))

          elif record.node == 'Oculomatic':
            x, y, diameter = 0.0, 0.0, 0.0
            for span in analog.spans:
              if span.name == 'X':
                x = analog.data[span.begin]
              elif span.name == 'Y':
                y = analog.data[span.begin]
              elif span.name == 'Diameter':
                diameter = analog.data[span.begin]
            current = oculomatic[record.time]
            oculomatic[record.time] = current._replace(x=x, y=y, diameter=diameter)

          elif record.node == 'Fiducial':
            # Fiducial node pulse events (may not be present if node wasn't recording)
            span = None
            for s in analog.spans:
              if s.name == '' and s.begin != s.end:
                span = s
            if span is not None:
              # AlexRig: no remote clock, so use analog.time for both
              fiducial_pulse.append(analog.time)
              fiducial.append(analog.time)

          elif record.node == 'Analog in':
            # NIDAQ data - AlexRig channel mapping:
            #   Dev1/ai0 = fiducial
            #   Dev1/ai1 = display
            #   Dev1/ai2 = reward (analog)
            if len(analog.sample_intervals) > 0:
              sample_interval = analog.sample_intervals[0] / 1e9
            for span in analog.spans:
              if span.name == FIDUCIAL_CHANNEL:
                fiducial_daq.extend(d > 2.5 for d in numpy.repeat(numpy.array(analog.data[span.begin:span.end]), upsample_factor))
              elif span.name == DISPLAY_CHANNEL:
                display_samples.extend(numpy.repeat(numpy.array(analog.data[span.begin:span.end]), upsample_factor))

          elif record.node == HAND_NODE:
            # Serial hand/touchscreen position data from Node 2 (raw screen coordinates).
            # Each record is one touch event (or hand-off (0,0)) at record.time.
            # It can be treated the same way as oculomatic (testing)
            x, y = 0.0, 0.0
            for span in analog.spans:
              if span.name == 'X':
                x = analog.data[span.begin]
              elif span.name == 'Y':
                y = analog.data[span.begin]
            current = hand[record.time]
            hand[record.time] = current._replace(x=x, y=y)

          elif record.node == 'Joystick':
            joystick.extend(_extract_joystick_samples(record.time, analog))

        if record.time != 0:
          time_range = min(time_range[0], record.time), max(time_range[1], record.time)
      rec_times.append(time_range)

    # has_hand_data = len(hand) > 0 #logic might need to be revised for oculomatic match
    # if not has_hand_data:
    #   print(f'WARNING: No touchscreen/hand data found for {rec.name}. Touch nodes may not be recording.')
    # else:
    #   print(f'Hand data: {len(hand)} touch events')

    has_fiducial_node = len(fiducial) > 0
    if not has_fiducial_node:
      print(f'WARNING: No Fiducial node records found for {rec.name}. Only DAQ fiducial signal available.')
    else:
      print(f'Fiducial node: {len(fiducial)} pulses found')

    array_dims = {k: numpy.array(v) for k, v in dims.items()}

    mat_dir = rec_dir / f'rec{rec_dir.name}.bag' / 'mat'
    mat_dir.mkdir(exist_ok=True, parents=True)

    if experiment_def_file is not None:
      shutil.copyfile(str(experiment_def_file), str(rec_dir / f'rec{rec_dir.name}.experiment.mat'))

    # --- Save behavioral data ---
    trial_summary_time = numpy.array(trial_summary_time)
    behav_state_time = numpy.array(behav_state_time)
    scipy.io.savemat(str(mat_dir / f'trial_summary.mat'), {
      'header_stamp_nanosec': trial_summary_time % 1_000_000_000,
      'header_stamp_sec': trial_summary_time // 1_000_000_000
    })
    scipy.io.savemat(str(mat_dir / f'state.mat'), {
      'header_stamp_nanosec': behav_state_time % 1_000_000_000,
      'header_stamp_sec': behav_state_time // 1_000_000_000
    })
    state_state_path = mat_dir / f'state_state.yaml'
    with open(state_state_path, 'w') as state_state_file:
      yaml.dump(behav_state, state_state_file)

    trial_summary_task_config_path = mat_dir / f'trial_summary_task_config.yaml'
    with open(trial_summary_task_config_path, 'w') as trial_summary_task_config_file:
      yaml.dump(task_config, trial_summary_task_config_file, default_style='"')

    trial_summary_task_result_path = mat_dir / f'trial_summary_task_result.yaml'
    with open(trial_summary_task_result_path, 'w') as trial_summary_task_result_file:
      yaml.dump(task_result, trial_summary_task_result_file, default_style='"')

    trial_summary_used_values_path = mat_dir / f'trial_summary_used_values.yaml'
    with open(trial_summary_used_values_path, 'w') as trial_summary_used_values_file:
      yaml.dump(used_values, trial_summary_used_values_file, default_style='"')

    trial_summary_behav_result_path = mat_dir / f'trial_summary_behav_result.yaml'
    with open(trial_summary_behav_result_path, 'w') as trial_summary_behav_result_file:
      yaml.dump(behave_result, trial_summary_behav_result_file, default_style='"')

    # --- Oculomatic ---
    oculomatic_pairs = sorted(oculomatic.items())
    if oculomatic_pairs:
      oculomatic_data = numpy.array([[o.x, o.y, o.width, o.height, o.diameter] for t, o in oculomatic_pairs])
    else:
      oculomatic_data = numpy.array([[], [], [], [], []]).T
    oculomatic_times = numpy.array([t for t, _ in oculomatic_pairs])
    scipy.io.savemat(str(mat_dir / f'oculomatic_eye.mat'), {
      'header_stamp_nanosec': oculomatic_times % 1_000_000_000,
      'header_stamp_sec': oculomatic_times // 1_000_000_000,
      'x': oculomatic_data[:, 0],
      'y': oculomatic_data[:, 1],
      'og_width': oculomatic_data[:, 2],
      'og_height': oculomatic_data[:, 3],
      'diameter': oculomatic_data[:, 4],
      'i': numpy.zeros_like(oculomatic_data[:, 0]),
      'width': numpy.zeros_like(oculomatic_data[:, 0]),
      'height': numpy.zeros_like(oculomatic_data[:, 0]),
      'step': numpy.zeros_like(oculomatic_data[:, 0])
    })
    encoding_path = mat_dir / f'oculomatic_eye_encoding.yaml'
    with open(encoding_path, 'w') as encoding_file:
      yaml.dump(['' for z in oculomatic_data[:, 0]], encoding_file)

    # --- Reward (from Reward node) ---
    reward_times = numpy.array([r.timestamp_ns for r in rewards])
    reward_durations_ns = numpy.array([r.on_time_ns for r in rewards])
    scipy.io.savemat(str(mat_dir / f'deliver_reward.mat'), {
      'header_stamp_nanosec': reward_times % 1_000_000_000,
      'header_stamp_sec': reward_times // 1_000_000_000,
      'on_time_ms': reward_durations_ns // 1_000_000,
    })

    scipy.io.savemat(str(mat_dir / f'recording_report.mat'), {
      'topic_time_stamp': 0
    })

    # --- Display data (from Analog in) ---
    display_samples = numpy.array(display_samples, dtype=numpy.float32)
    display_path = rec_dir / f'rec{rec_number}.display.dat'
    display_samples.tofile(str(display_path))
    print(f'Display: {len(display_samples)} samples written to {display_path}')

    # --- Hand data (from touchscreen) ---
    # Serial hand data is treated similarly to oculomatic
    #if has_hand_data:
    hand_pairs = sorted(hand.items())
    if hand_pairs:
        hand_data = numpy.array([[h.x, h.y] for t, h in hand_pairs])
    else:
        hand_data = numpy.array([[], [], [], [], []]).T
    hand_times = numpy.array([t for t, _ in hand_pairs])
    scipy.io.savemat(str(mat_dir / f'serialhnd.mat'), {
    'header_stamp_nanosec': hand_times % 1_000_000_000,
    'header_stamp_sec': hand_times // 1_000_000_000,
    'x': hand_data[:, 0],
    'y': hand_data[:, 1]
    })
    # else:
    #   # Write empty file as placeholder
    #   numpy.array([], dtype=numpy.float32).tofile(str(hand_path))
    #   print(f'Hand: empty placeholder written to {hand_path}')

    # --- Joystick data ---
    joystick.sort(key=lambda sample: sample.timestamp_ns)
    if joystick:
        joystick_data = numpy.array([[j.x, j.y] for j in joystick], dtype=float)
        joystick_times = numpy.array([j.timestamp_ns for j in joystick], dtype=numpy.int64)
    else:
        joystick_data = numpy.array([[], []]).T
        joystick_times = numpy.array([], dtype=numpy.int64)
    scipy.io.savemat(str(mat_dir / 'joystick.mat'), {
      'header_stamp_nanosec': joystick_times % 1_000_000_000,
      'header_stamp_sec': joystick_times // 1_000_000_000,
      'x': joystick_data[:, 0] if len(joystick_data) > 0 else numpy.array([]),
      'y': joystick_data[:, 1] if len(joystick_data) > 0 else numpy.array([]),
    })
    print(f'Joystick: {len(joystick)} samples')

    # --- Fiducial data ---
    residual_outliers = etl_config.get("residual_outliers", {}).get(str(int(rec_dir.name)), [])
    print(f'residual_outliers: {residual_outliers}')

    fiducial = numpy.array(fiducial, dtype=numpy.uint64)
    fiducial_pulse = numpy.array(fiducial_pulse, dtype=numpy.uint64)
    fiducial_daq = numpy.array(fiducial_daq, dtype=bool)

    fiducial = numpy.delete(fiducial, residual_outliers)
    fiducial_pulse = numpy.delete(fiducial_pulse, residual_outliers)
    if residual_outliers:
      edges, = numpy.where(numpy.diff(fiducial_daq))
      for r in residual_outliers:
        begin, end = 2 * r, 2 * r + 1
        if end < len(edges):
          begin_edge, end_edge = edges[begin], edges[end]
          fiducial_daq[begin_edge:end_edge] = False

    # Save fiducial node data (may be empty if node wasn't recording)
    scipy.io.savemat(str(mat_dir / 'fiducial.mat'), {
      'time_ref_nanosec': fiducial % 1_000_000_000 if len(fiducial) > 0 else numpy.array([]),
      'time_ref_sec': fiducial // 1_000_000_000 if len(fiducial) > 0 else numpy.array([]),
      'topic_time_stamp': fiducial if len(fiducial) > 0 else numpy.array([])
    })
    scipy.io.savemat(str(mat_dir / 'fiducial_pulse.mat'), {
      'stamp_nanosec': fiducial_pulse % 1_000_000_000 if len(fiducial_pulse) > 0 else numpy.array([]),
      'stamp_sec': fiducial_pulse // 1_000_000_000 if len(fiducial_pulse) > 0 else numpy.array([]),
      'topic_time_stamp': fiducial_pulse if len(fiducial_pulse) > 0 else numpy.array([])
    })

    # Save raw DAQ fiducial signal
    scipy.io.savemat(str(rec_dir / f'rec{rec_dir.name}.fiducial.mat'), {
      'fiducial': fiducial_daq
    })

    source_path = mat_dir / 'fiducial_source.yaml'
    with open(source_path, 'w') as source_file:
      yaml.dump(['fiducial_node' for t in fiducial], source_file)

    frame_path = mat_dir / 'fiducial_header_frame_id.yaml'
    with open(frame_path, 'w') as frame_file:
      yaml.dump(['' for t in fiducial], frame_file)

    frame_path = mat_dir / 'fiducial_pulse_frame_id.yaml'
    with open(frame_path, 'w') as frame_file:
      yaml.dump(['' for t in fiducial], frame_file)

    # --- Recorder meta ---
    print(rec_dir.name, sample_interval, 1.0 / sample_interval if sample_interval != sys.maxsize else 'unknown')
    recorder_meta_path = rec_dir / f'rec{rec_dir.name}.recorder_meta.mat'
    actual_fs = 1.0 / sample_interval if sample_interval != sys.maxsize else nidaq_sample_rate
    scipy.io.savemat(str(recorder_meta_path), {
      'rec_meta': numpy.array([(actual_fs, 'single')], dtype=REC_META_TYPE)
    })

    # --- Video (already extracted for AlexRig) ---
    img_dir = rec_dir / f'rec{rec_dir.name}.bag' / 'img'
    topics = []

    # Collect image node names from image_times (since mux=False, record_reader.image_nodes is empty)
    video_nodes = [n for n in image_times.keys() if n != 'Oculomatic']
    for node in video_nodes:
      # Skip if outputs are already complete (MP4 symlink + compressed image mat both present)
      if args.skip_existing_video and \
         (img_dir / node / 'image.mp4').exists() and \
         (mat_dir / f'{node}_image_raw_compressed.mat').exists():
        print(f'Video: {node} already complete, skipping')
        continue

      # Look for pre-extracted MP4 files
      # AlexRig naming: "rec{N}.{node}.mp4" or "{node}.mp4"
      mp4_candidates = [
        day_dir / f'rec{rec_int}.{node}.mp4',
        day_dir / f'{node}.mp4',
      ]
      mp4_file = None
      for candidate in mp4_candidates:
        if candidate.exists():
          mp4_file = candidate
          break

      times = numpy.array(image_times[node], dtype=numpy.uint64)

      if mp4_file is not None:
        image_path = img_dir / node / 'image.mp4'
        image_path.parent.mkdir(exist_ok=True, parents=True)
        image_path.unlink(missing_ok=True)
        # Prefer a symlink to preserve the original file, but fall back to copy
        # on Windows systems without symlink privileges.
        try:
          image_path.symlink_to(mp4_file.resolve())
          print(f'Video: linked {mp4_file} -> {image_path}')
        except OSError:
          shutil.copyfile(str(mp4_file), str(image_path))
          print(f'Video: copied {mp4_file} -> {image_path}')
      else:
        print(f'WARNING: No MP4 file found for {node} rec {rec_int}')

      frame_ids = [node for t in times]
      formats = ['mono8; mpeg4 compressed' for t in times]
      format_path = mat_dir / f'{node}_image_raw_compressed_format.yaml'
      frame_id_path = mat_dir / f'{node}_image_raw_compressed_header_frame_id.yaml'
      with open(frame_id_path, 'w') as frame_id_file:
        yaml.dump(frame_ids, frame_id_file)
      with open(format_path, 'w') as format_file:
        yaml.dump(formats, format_file)
      frame_id_ords = numpy.array([ord(c) for c in str(frame_id_path)])
      format_ords = numpy.array([ord(c) for c in str(format_path)])
      scipy.io.savemat(str(mat_dir / f'{node}_image_raw_compressed.mat'), {
        'header_stamp_nanosec': times % 1_000_000_000,
        'header_stamp_sec': times // 1_000_000_000,
        'topic_time_stamp': times,
        'format': format_ords,
        'header_frame_id': frame_id_ords
      })

      topics.append({
        'topic_metadata': {
          'name': f'/{node}/image_raw/compressed',
          'type': 'sensor_msgs/msg/CompressedImage',
          'serialization_format': 'cdr'
        },
        'message_count': 1
      })

      if behave_config is not None:
        for config_node in behave_config['nodes']:
          if config_node['type'] != 'DISTORTION':
            continue
          if config_node.get('Source') != node:
            continue
          d = numpy.array(config_node['Distortion Coefficients'])
          k = numpy.array(config_node['Camera Matrix'])

          p = numpy.zeros((3, 4))
          p[:3, :3] = k
          p = p.flatten()[numpy.newaxis].repeat(len(times), axis=0)

          d = d[numpy.newaxis].repeat(len(times), axis=0)
          k = k.flatten()
          k = k[numpy.newaxis].repeat(len(times), axis=0)
          r = numpy.eye(3).flatten()[numpy.newaxis].repeat(len(times), axis=0)
          frame_id_path = mat_dir / f'{node}_camera_info_header_frame_id.yaml'
          with open(frame_id_path, 'w') as frame_id_file:
            yaml.dump(frame_ids, frame_id_file)
          distortion_model = ['plumb_bob' for t in times]
          distortion_model_path = mat_dir / f'{node}_camera_info_distortion_model.yaml'
          with open(distortion_model_path, 'w') as distortion_model_file:
            yaml.dump(distortion_model, distortion_model_file)
          frame_id_ords = numpy.array([ord(c) for c in str(frame_id_path)])
          distortion_model_ords = numpy.array([ord(c) for c in str(distortion_model_path)])
          camera_info_dims = array_dims[node]
          width = camera_info_dims[:, 0]
          height = camera_info_dims[:, 1]

          scipy.io.savemat(str(mat_dir / f'{node}_camera_info.mat'), {
            'header_stamp_nanosec': times % 1_000_000_000,
            'header_stamp_sec': times // 1_000_000_000,
            'topic_time_stamp': times,
            'k': k,
            'd': d,
            'r': r,
            'p': p,
            'binning_x': numpy.ones(len(times)),
            'binning_y': numpy.ones(len(times)),
            'distortion_model': distortion_model_ords,
            'header_frame_id': frame_id_ords,
            'width': width,
            'height': height,
            'roi_do_rectify': numpy.zeros(len(times)),
            'roi_width': numpy.zeros(len(times)),
            'roi_height': numpy.zeros(len(times)),
            'roi_x_offset': numpy.zeros(len(times)),
            'roi_y_offset': numpy.zeros(len(times))
          })
          topics.append({
            'topic_metadata': {
              'name': f'/{node}/camera_info',
              'type': 'sensor_msgs/msg/CameraInfo',
              'serialization_format': 'cdr'
            },
            'message_count': 1
          })
          break

    metadata = {
      'rosbag2_bagfile_information': {
        'version': 2,
        'storage_identifier': 'sqlite3',
        'relative_file_paths': [str(rec.absolute())],
        'duration': {'nanoseconds': time_range[1] - time_range[0]},
        'starting_time': {'nanoseconds_since_epoch': int(wall_time)},
        'message_count': message_count,
        'topics_with_message_count': topics
      }
    }
    with open(mat_dir.parent / 'metadata.yaml', 'w') as metadata_file:
      yaml.dump(metadata, metadata_file)

  # --- Summary ---
  for r, r2 in zip(rec_times, recs):
    print(r2, (r[1] - r[0]) / 1e9)
  print(recs)
  print(rec_times)

if __name__ == '__main__':
  main()
