# One tracked object (output of scripts/object_tracker.py).
# Position/velocity are in header.frame_id of the TrackArray (base_link: relative to the ego car,
# velocity = object's velocity over ground, expressed in base_link axes).
uint32 id
geometry_msgs/Point position        # centre of the object footprint, z = top height above road
geometry_msgs/Vector3 velocity      # m/s over ground
geometry_msgs/Vector3 size          # footprint length (x), width (y), height (z), axis-aligned
float32 speed                       # |velocity|, m/s
bool moving                         # confirmed and faster than the tracker's moving threshold
uint16 age                          # frames since the track was created
uint16 hits                         # frames with a matching detection
string label                        # camera class from object_detector ('' = unknown): person, car, ...
float32 label_score                 # detector confidence of that label
