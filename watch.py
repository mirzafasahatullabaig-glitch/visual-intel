"""Turn tracker output into plain-language events and anomaly alerts.

Events (normal):  'person #1 appeared at centre', 'person #1 left; last seen at right'
Alerts (unusual):
  - an object that had been in view for at least SETTLED seconds disappears
  - a person enters the restricted zone (drawn with the mouse in main.py)
  - an object is left unattended: it arrived while the program was running
    and has now sat there with nobody in view for UNATTENDED seconds
Objects seen only briefly never raise alerts: they are usually detector mistakes.
Not covered: detecting that an object FELL (boxes alone can't tell a fall
from someone moving the object down; that needs a pose/motion model).
"""
import time

import track

UNATTENDED = 20  # seconds with nobody in view before a newly arrived object is flagged
STARTUP = 5      # objects already present in the first seconds are background, not 'left behind'
SETTLED = 10     # seconds an object must have been in view before its disappearance is an alert


class Watcher:
    def __init__(self, now=None):
        self.start = self.last_person = time.time() if now is None else now

    def check(self, visible, events, zone, now=None):
        """visible: tracks in view; events: from Tracker.update; zone: (x1, y1, x2, y2) or None.
        Returns [(text, is_alert), ...]."""
        now = time.time() if now is None else now
        out = []
        for kind, t in events:
            place = f"at {t['where']}" + (f" (near {t['near']})" if t["near"] else "")
            if kind == "entered":
                out.append((f"{track.label(t)} appeared {place}", False))
            elif t["name"] == "person" or t["last"] - t["first"] < SETTLED:
                # People come and go; and something seen only briefly is usually a detector mistake.
                out.append((f"{track.label(t)} left; last seen {place}", False))
            else:
                out.append((f"{track.label(t)} disappeared; last seen {place}", True))

        people = [t for t in visible if t["name"] == "person"]
        if people:
            self.last_person = now
        for t in people:  # restricted zone: alert once each time a person's centre goes in
            cx, cy = track.centre(t["box"])
            inside = bool(zone) and zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]
            if inside and not t.get("in_zone"):
                out.append((f"{track.label(t)} entered the restricted zone", True))
            t["in_zone"] = inside

        if now - self.last_person > UNATTENDED:
            for t in visible:
                arrived_later = t["first"] > self.start + STARTUP
                stayed = now - t["first"] > UNATTENDED  # a 2-second detector mistake is not an abandoned object
                if t["name"] != "person" and arrived_later and stayed and not t.get("flagged"):
                    t["flagged"] = True
                    out.append((f"{track.label(t)} left unattended at {t['where']}", True))
        return out


if __name__ == "__main__":
    # Self-check with a fake clock: a person brings a bag, walks into the zone, then leaves the bag.
    tr, w, t0 = track.Tracker(), Watcher(now=0.0), 0.0
    log, zone = [], (400, 0, 640, 480)

    def step(dets, now):
        events = tr.update(dets, 640, 480, now=now)
        log.extend(text for text, alert in w.check(tr.visible(now), events, zone, now=now) if alert)

    for i in range(30):   # 0-1 s: person alone on the left
        step([("person", 0.9, [50, 100, 150, 300])], i / 30)
    # 10-11 s: after a gap the person is back (so a new ID, #2) inside the zone, with a backpack (#3)
    for i in range(30):
        step([("person", 0.9, [420, 100, 150, 300]), ("backpack", 0.8, [100, 300, 80, 80])], 10 + i / 30)
    assert log == ["person #2 entered the restricted zone"], log
    for i in range(30 * 25):  # 11-36 s: person gone, backpack stays
        step([("backpack", 0.8, [100, 300, 80, 80])], 11 + i / 30)
    assert log[-1] == "backpack #3 left unattended at bottom-left" and len(log) == 2, log
    step([], 40.0)        # backpack vanishes
    assert log[-1] == "backpack #3 disappeared; last seen at bottom-left", log
    # Something seen for only 2 s (a detector mistake) with nobody around must raise no alert at all.
    for i in range(60):
        step([("refrigerator", 0.5, [300, 100, 100, 200])], 50 + i / 30)
    step([], 60.0)
    assert len(log) == 3, log[3:]
    print("\n".join(log))
    print("OK: alerts work.")
