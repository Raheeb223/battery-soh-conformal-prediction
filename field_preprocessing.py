"""
Field check, step 1: the EVBattery vehicles (evbattery_loader.py) turned into the same kind of cache
as the laboratory data, so every later step can reuse the laboratory code unchanged.

Mapping. Each vehicle plays the role of a cell and each labelled charging event the role of a cycle.
An event's capacity label is the same for every snippet of that event, so each event becomes one
point. Events are ordered by mileage. SOH is the capacity divided by the median of the vehicle's
first three labelled events, the same definition as for laboratory cells (preprocessing.add_soh).
The vehicles enter the data part-way through their life, so this SOH is the capacity relative to the
vehicle's first labelled events, not to a new battery. The three manufacturers are the three groups
("EV1", "EV2", "EV3"), each normalized with its own training vehicles.

The vehicle-level 70/15/15 split, the 20-event windows and the normalization are preprocessing.py's
own functions (cell_independent_split, make_windows, run), unchanged.

Requires: EVBattery/battery_dataset{1,2,3}.tar.gz (figshare DOI 10.6084/m9.figshare.23301881)
Output:   cache/field_processed.npz (cache/field_processed_split{N}.npz for other split seeds)
Run:      python field_preprocessing.py
"""
import os

import config
import evbattery_loader
import preprocessing


def load_events():
    rows = evbattery_loader.build_rows()
    events = (rows.groupby(["manufacturer", "vehicle", "charge_segment"])
              .agg(mileage=("mileage", "median"), capacity=("capacity_ah", "median"),
                   label_spread=("capacity_ah", lambda s: s.max() - s.min()))
              .reset_index())
    assert (events["label_spread"] == 0).all(), "snippets of one charging event carry different labels"
    events = events.sort_values(["vehicle", "mileage", "charge_segment"])
    events["cycle"] = events.groupby("vehicle").cumcount() + 1
    return events.rename(columns={"manufacturer": "dataset", "vehicle": "cell_id"})[
        ["dataset", "cell_id", "cycle", "capacity"]]


def cache_path(split_seed=None):
    default = getattr(config, "SPLIT_SEED", config.RANDOM_SEED)
    suffix = "" if split_seed in (None, default) else f"_split{split_seed}"
    return os.path.join(config.CACHE_DIR, f"field_processed{suffix}.npz")


def run(split_seed=None, df=None):
    df = df if df is not None else preprocessing.add_soh(load_events())
    return preprocessing.run(split_seed=split_seed, raw_df=df, out_path=cache_path(split_seed))


if __name__ == "__main__":
    run()
