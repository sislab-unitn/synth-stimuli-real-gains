from collections import Counter
import json
import os
import argparse

from statistics import mean, stdev
from itertools import chain, product
from PIL import Image
from tqdm import tqdm
from world.questions import PROP_VALUES, absolute_position_question
from world.sprites import SpritesLoader
from world.utils import SIZES, random_value_order
from world.world import World


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--size", type=int, default=9)
    args = parser.parse_args()

    os.makedirs(args.name, exist_ok=True)

    world_size = [args.size, args.size]

    sprites_loader = SpritesLoader()

    worlds_repr = {}
    questions = {}

    positions = [(i,j) for i in range(args.size) for j in range(args.size)]

    # property variations for one object
    shapes = ["square", "circle", "triangle", "star"]
    colors = ["red", "green", "blue", "magenta", "yellow", "cyan"]
    # all shapes, white
    prop_vairations =  list(product(shapes, ["white"], SIZES, positions))
    # plus, all colors
    prop_vairations += list(product(["plus"], colors, SIZES, positions))


    n_stimuli = len(prop_vairations)

    n_shapes = len(PROP_VALUES["shape"])
    n_colors = len(PROP_VALUES["color"])
    shape_orders = random_value_order(n_stimuli, n_shapes)
    color_orders = random_value_order(n_stimuli, n_colors)

    n_areas = len(PROP_VALUES["area"])
    n_areas_vert = len(PROP_VALUES["area_vert"])
    n_area_hors = len(PROP_VALUES["area_hor"])
    area_orders = random_value_order(n_stimuli, n_areas)
    area_vert_orders = random_value_order(n_stimuli, n_areas_vert)
    area_hor_orders = random_value_order(n_stimuli, n_area_hors)

    for name, orders in zip(
        ["shape", "color", "area", "area_vert", "area_hor"],
        [shape_orders, color_orders, area_orders, area_vert_orders, area_hor_orders]
    ):
        print("-"*40)
        print(name)
        c = Counter(orders)
        print(f"{mean(c.values())} +- {stdev(c.values())}, max-min: {max(c.values()) - min(c.values())}")


    variations = []
    for v, shape_ord, color_ord, area_ord, area_vert_ord, area_hor_ord in zip(prop_vairations, shape_orders, color_orders, area_orders, area_vert_orders, area_hor_orders):
        variation = [v, shape_ord, color_ord]
        variation.extend([area_ord, area_vert_ord, area_hor_ord])
        variations.append(tuple(chain(variation)))

    assert len(variations) == n_stimuli

    for stim_n, (variation, shape_ord, color_ord, area_ord, area_vert_ord, area_hor_ord) in tqdm(enumerate(variations), desc=f"Generating stimuli", unit="stimuli", total=len(variations)):

        params = {
            "size": world_size,
            "img_size": args.img_size,
            "background": "none",
            "objects": [],
        }

        shape, color, size, position = variation
        params["objects"].append({
            "count": 1,
            "shape": shape,
            "color": color,
            "sheen": "none",
            "size": size,
            "position": position
        })

        grid_size = tuple(params["size"])

        background = "none"
        if "background" in params:
            background = params["background"]

        objects = params["objects"]

        world = World(grid_size, background=background)

        for obj_template in objects:
            world.add(**obj_template)

        s = world.get_stimulus(sprites_loader, img_size=args.img_size)
        save_folder = os.path.join(args.name, "images")
        save_path = os.path.join(save_folder, f"{stim_n}.png")
        os.makedirs(save_folder, exist_ok=True)
        Image.fromarray(s).save(save_path)

        worlds_repr[stim_n] = world.to_dict(img_path=save_path)
        e = world.ents["e1"]
        if stim_n not in questions:
            questions[stim_n] = {}
        if e.id not in questions[stim_n]:
            questions[stim_n][e.id] = {}
        questions[stim_n][e.id]["position_absolute"] = absolute_position_question(e, val_order=area_ord)

    with open(os.path.join(args.name, "repr.json"), "w") as f:
        json.dump(worlds_repr, f, indent=4)
    with open(os.path.join(args.name, "questions.json"), "w") as f:
        json.dump(questions, f, indent=4)

if __name__ == "__main__":
    main()
