import os
import sys
from nltk.stem import *
import nltk
import json
import argparse
# from .misc import *
import re
lemma = nltk.wordnet.WordNetLemmatizer()

#### SINGULARIZE #########################################################
# Adapted from Bermi Ferrer's Inflector for Python:
# http://www.bermi.org/inflector/

# Copyright (c) 2006 Bermi Ferrer Martinez
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software to deal in this software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of this software, and to permit
# persons to whom this software is furnished to do so, subject to the following
# condition:
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THIS SOFTWARE.

_singular_rules = [
    (r'(?i)(.)ae$', '\\1a'),
    (r'(?i)(.)itis$', '\\1itis'),
    (r'(?i)(.)eaux$', '\\1eau'),
    (r'(?i)(quiz)zes$', '\\1'),
    (r'(?i)(matr)ices$', '\\1ix'),
    (r'(?i)(ap|vert|ind)ices$', '\\1ex'),
    (r'(?i)^(ox)en', '\\1'),
    (r'(?i)(alias|status)es$', '\\1'),
    (r'(?i)([octop|vir])i$',  '\\1us'),
    (r'(?i)(cris|ax|test)es$', '\\1is'),
    (r'(?i)(shoe)s$', '\\1'),
    (r'(?i)(o)es$', '\\1'),
    (r'(?i)(bus)es$', '\\1'),
    (r'(?i)([m|l])ice$', '\\1ouse'),
    (r'(?i)(x|ch|ss|sh)es$', '\\1'),
    (r'(?i)(m)ovies$', '\\1ovie'),
    (r'(?i)(.)ombies$', '\\1ombie'),
    (r'(?i)(s)eries$', '\\1eries'),
    (r'(?i)([^aeiouy]|qu)ies$', '\\1y'),
    # -f, -fe sometimes take -ves in the plural
    # (e.g., lives, wolves).
    (r"([aeo]l)ves$", "\\1f"),
    (r"([^d]ea)ves$", "\\1f"),
    (r"arves$", "arf"),
    (r"erves$", "erve"),
    (r"([nlw]i)ves$", "\\1fe"),
    (r'(?i)([lr])ves$', '\\1f'),
    (r"([aeo])ves$", "\\1ve"),
    (r'(?i)(sive)s$', '\\1'),
    (r'(?i)(tive)s$', '\\1'),
    (r'(?i)(hive)s$', '\\1'),
    (r'(?i)([^f])ves$', '\\1fe'),
    # -ses suffixes.
    (r'(?i)(^analy)ses$', '\\1sis'),
    (r'(?i)((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$',
     '\\1\\2sis'),
    (r'(?i)(.)opses$', '\\1opsis'),
    (r'(?i)(.)yses$', '\\1ysis'),
    (r'(?i)(h|d|r|o|n|b|cl|p)oses$', '\\1ose'),
    (r'(?i)(fruct|gluc|galact|lact|ket|malt|rib|sacchar|cellul)ose$',
     '\\1ose'),
    (r'(?i)(.)oses$', '\\1osis'),
    # -a
    (r'(?i)([ti])a$', '\\1um'),
    (r'(?i)(n)ews$', '\\1ews'),
    (r'(?i)([^s])s$', '\\1'),  # don't make ss singularize to s.
]

# For performance, compile the regular expressions only once:
_singular_rules = [(re.compile(r[0]), r[1]) for r in _singular_rules]

_singular_uninflected = set((
    "bison", "debris", "headquarters", "pincers", "trout",
    "bream", "diabetes", "herpes", "pliers", "tuna",
    "breeches", "djinn", "high-jinks", "proceedings", "whiting",
    "britches", "eland", "homework", "rabies", "wildebeest"
    "carp", "elk", "innings", "salmon",
    "chassis", "flounder", "jackanapes", "scissors",
    "christmas", "gallows", "mackerel", "series",
    "clippers", "georgia", "measles", "shears",
    "cod", "graffiti", "mews", "species",
    "contretemps", "mumps", "swine",
    "corps", "news", "swiss",
    # Custom added from MD&A corpus
    "api", "mae", "sae", "basis", "india", "media",
))
_singular_uncountable = set((
    "advice", "equipment", "happiness", "luggage", "news", "software",
    "bread", "fruit", "information", "mathematics", "progress", "understanding",
    "butter", "furniture", "ketchup", "mayonnaise", "research", "water"
    "cheese", "garbage", "knowledge", "meat", "rice",
    "electricity", "gravel", "love", "mustard", "sand",
))
_singular_ie = set((
    "alergie", "cutie", "hoagie", "newbie", "softie", "veggie",
    "auntie", "doggie", "hottie", "nightie", "sortie", "weenie",
    "beanie", "eyrie", "indie", "oldie", "stoolie", "yuppie",
    "birdie", "freebie", "junkie", "^pie", "sweetie", "zombie"
    "bogie", "goonie", "laddie", "pixie", "techie",
    "bombie", "groupie", "laramie", "quickie", "^tie",
    "collie", "hankie", "lingerie", "reverie", "toughie",
    "cookie", "hippie", "meanie", "rookie", "valkyrie",
))
_singular_irregular = {
    "abuses": "abuse",
    "ads": "ad",
    "atlantes": "atlas",
    "atlases": "atlas",
    "analysis": "analysis",
    "axes": "axe",
    "beeves": "beef",
    "brethren": "brother",
    "children": "child",
    "children": "child",
    "corpora": "corpus",
    "corpuses": "corpus",
    "ephemerides": "ephemeris",
    "feet": "foot",
    "ganglia": "ganglion",
    "geese": "goose",
    "genera": "genus",
    "genii": "genie",
    "graffiti": "graffito",
    "helves": "helve",
    "kine": "cow",
    "leaves": "leaf",
    "loaves": "loaf",
    "men": "man",
    "mongooses": "mongoose",
    "monies": "money",
    "moves": "move",
    "mythoi": "mythos",
    "numena": "numen",
    "occipita": "occiput",
    "octopodes": "octopus",
    "opera": "opus",
    "opuses": "opus",
    "our": "my",
    "oxen": "ox",
    "penes": "penis",
    "penises": "penis",
    "people": "person",
    "sexes": "sex",
    "soliloquies": "soliloquy",
    "teeth": "tooth",
    "testes": "testis",
    "trilbys": "trilby",
    "turves": "turf",
    "zoa": "zoon",
}

_plural_prepositions = set((
    "about", "before", "during", "of", "till",
    "above", "behind", "except", "off", "to",
    "across", "below", "for", "on", "under",
    "after", "beneath", "from", "onto", "until",
    "among", "beside", "in", "out", "unto",
    "around", "besides", "into", "over", "upon",
    "at", "between", "near", "since", "with",
    "athwart", "betwixt", "beyond", "but", "by"
))

import re
 
def singularize(word, custom={}):
    """Returns the singular of a given word."""
    if word in custom:
        return custom[word]
    # Recurse compound words (e.g. mothers-in-law).
    if "-" in word:
        w = word.split("-")
        if len(w) > 1 and w[1] in _plural_prepositions:
            return singularize(w[0], custom) + "-" + "-".join(w[1:])
    # dogs' => dog's
    if word.endswith("'"):
        return singularize(word[:-1], custom) + "'s"
    w = word.lower()
    for x in _singular_uninflected:
        if x.endswith(w):
            return word
    for x in _singular_uncountable:
        if x.endswith(w):
            return word
    for x in _singular_ie:
        if w.endswith(x + "s"):
            return w
    for x in _singular_irregular:
        if w.endswith(x):
            return re.sub('(?i)' + x + '$', _singular_irregular[x], word)
    for suffix, inflection in _singular_rules:
        m = suffix.search(word)
        g = m and m.groups() or []
        if m:
            for k in range(len(g)):
                if g[k] is None:
                    inflection = inflection.replace('\\' + str(k + 1), '')
            return suffix.sub(inflection, word)
    return word


def combine_coco_captions(annotation_path):
    if not os.path.exists("%s/captions_%s2014.json" % (annotation_path, "val")):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists("%s/captions_%s2014.json" % (annotation_path, "train")):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open("%s/captions_%s2014.json" % (annotation_path, "val")))
    train_caps = json.load(open("%s/captions_%s2014.json" % (annotation_path, "train")))
    all_caps = {
        "info": train_caps["info"],
        "licenses": train_caps["licenses"],
        "images": val_caps["images"] + train_caps["images"],
        "annotations": val_caps["annotations"] + train_caps["annotations"],
    }

    return all_caps


def combine_coco_instances(annotation_path):
    if not os.path.exists("%s/instances_%s2014.json" % (annotation_path, "val")):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists("%s/instances_%s2014.json" % (annotation_path, "train")):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(
        open("%s/instances_%s2014.json" % (annotation_path, "val"))
    )
    train_instances = json.load(
        open("%s/instances_%s2014.json" % (annotation_path, "train"))
    )
    all_instances = {
        "info": train_instances["info"],
        "licenses": train_instances["licenses"],
        "type": train_instances["licenses"],
        "categories": train_instances["categories"],
        "images": train_instances["images"] + val_instances["images"],
        "annotations": val_instances["annotations"] + train_instances["annotations"],
    }

    return all_instances


class CHAIR(object):
    def __init__(self, imids, coco_path):
        self.imid_to_objects = {imid: [] for imid in imids}
        self.imids = imids
        self.coco_path = coco_path

        # read in synonyms
        synonyms = open("./eval_scripts/eval_utils/data/synonyms.txt").readlines()
        synonyms = [s.strip().split(", ") for s in synonyms]
        self.mscoco_objects = []  # mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # Some hard coded rules for implementing CHAIR metrics on MSCOCO

        # common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = [
            "motor bike",
            "motor cycle",
            "air plane",
            "traffic light",
            "street light",
            "traffic signal",
            "stop light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "suit case",
            "sports ball",
            "baseball bat",
            "baseball glove",
            "tennis racket",
            "wine glass",
            "hot dog",
            "cell phone",
            "mobile phone",
            "teddy bear",
            "hair drier",
            "potted plant",
            "bow tie",
            "laptop computer",
            "stove top oven",
            "hot dog",
            "teddy bear",
            "home plate",
            "train track",
        ]

        # Hard code some rules for special cases in MSCOCO
        # qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = [
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "animal",
            "cub",
        ]
        # qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ["jet", "train"]

        # double_word_dict will map double words to the word they should be treated as in our analysis

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict["baby %s" % animal_word] = animal_word
            self.double_word_dict["adult %s" % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict["passenger %s" % vehicle_word] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

    def _load_generated_captions_into_evaluator(self, cap_file):
        """
        Meant to save time so imid_to_objects does not always need to be recomputed.
        """
        # Read in captions
        self.caps, imids, self.metrics = load_generated_captions(cap_file)

        assert imids == set(self.imid_to_objects.keys())

    def caption_to_words(self, caption):
        """
        Input: caption
        Output: MSCOCO words in the caption
        """

        # standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        # words = [singularize(w) for w in words]
        # replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i : i + 2])
            if singularize(double_word) in self.double_word_dict:
                double_words.append(self.double_word_dict[singularize(double_word)])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ("toilet" in words) & ("seat" in words):
            words = [word for word in words if word != "seat"]

        # get synonyms for all words in the caption
        idxs = [
            idxs[idx]
            for idx, word in enumerate(words)
            if singularize(word) in set(self.mscoco_objects)
        ]
        words = [word for word in words if singularize(word) in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[singularize(word)])
        # return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        """
        Add objects taken from MSCOCO segmentation masks
        """

        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments["annotations"]
        segment_annotations = [seg_anno for seg_anno in segment_annotations if seg_anno['image_id'] in self.imids]

        # make dict linking object name to ids
        id_to_name = {}  # dict with id to synsets
        for cat in coco_segments["categories"]:
            id_to_name[cat["id"]] = cat["name"]

        for i, annotation in enumerate(segment_annotations):
            sys.stdout.write(
                "\rGetting annotations for %d/%d segmentation masks"
                % (i+1, len(segment_annotations))
            )
            imid = annotation["image_id"]
            if imid in self.imid_to_objects:
                node_word = self.inverse_synonym_dict[
                    id_to_name[annotation["category_id"]]
                ]
                self.imid_to_objects[imid].append(node_word)
        print("\n")
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def get_annotations_from_captions(self):
        """
        Add objects taken from MSCOCO ground truth captions
        """

        coco_caps = combine_coco_captions(self.coco_path)
        caption_annotations = coco_caps["annotations"]
        caption_annotations = [cap_anno for cap_anno in caption_annotations if cap_anno['image_id'] in self.imids]

        for i, annotation in enumerate(caption_annotations):
            sys.stdout.write(
                "\rGetting annotations for %d/%d ground truth captions"
                % (i+1, len(caption_annotations))
            )
            imid = annotation["image_id"]
            if imid in self.imid_to_objects:
                _, node_words, _, _ = self.caption_to_words(annotation["caption"])
                self.imid_to_objects[imid].update(node_words)
        print("\n")

        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def get_annotations(self):
        """
        Get annotations from both segmentation and captions.  Need both annotation types for CHAIR metric.
        """

        self.get_annotations_from_segments()
        self.get_annotations_from_captions()

    def compute_chair(self, caps):
        """
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.
        """

        # self._load_generated_captions_into_evaluator(cap_file)
        self.caps = caps
        imid_to_objects = self.imid_to_objects
        caps = self.caps

        num_caps = 0.0
        num_hallucinated_caps = 0.0
        hallucinated_word_count = 0.0
        coco_word_count = 0.0

        output = {"sentences": []}

        for i, cap_eval in enumerate(caps):
            cap = cap_eval["caption"]
            imid = cap_eval["image_id"]

            # get all words in the caption, as well as corresponding node word
            words, node_words, idxs, raw_words = self.caption_to_words(cap)

            gt_objects = imid_to_objects[imid]
            cap_dict = {
                "image_id": cap_eval["image_id"],
                "image": cap_eval["image"],
                "caption": cap,
                "mscoco_hallucinated_words": [],
                "mscoco_non_hallucinated_words": [],
                "mscoco_gt_words": list(gt_objects),
                "mscoco_generated_words": list(words), 
                "hallucination_idxs": [],
                "non_hallucination_idxs": [],
                # "words": raw_words,
            }
            cap_dict["metrics"] = {
                "CHAIRs": 0,
                "CHAIRi": 0,
            }

            # count hallucinated words
            coco_word_count += len(node_words)
            hallucinated = False
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1
                    cap_dict["mscoco_hallucinated_words"].append((word, node_word))
                    cap_dict["hallucination_idxs"].append(idx)
                    hallucinated = True
                else:
                    cap_dict["mscoco_non_hallucinated_words"].append((word, node_word))
                    cap_dict["non_hallucination_idxs"].append(idx)

            # count hallucinated caps
            num_caps += 1
            if hallucinated:
                num_hallucinated_caps += 1

            cap_dict["metrics"]["CHAIRs"] = int(hallucinated)
            cap_dict["metrics"]["CHAIRi"] = 0.0
            if len(words) > 0:
                cap_dict["metrics"]["CHAIRi"] = len(
                    cap_dict["mscoco_hallucinated_words"]
                ) / float(len(words))

            output["sentences"].append(cap_dict)

        chair_s = num_hallucinated_caps / num_caps
        chair_i = hallucinated_word_count / coco_word_count

        output["overall_metrics"] = {
            "CHAIRs": chair_s,
            "CHAIRi": chair_i,
        }

        return output


def load_generated_captions(cap_file):
    # Read in captions, eg.
    caps = json.load(open(cap_file))
    try:
        metrics = caps["overall"]
        caps = caps["imgToEval"].values()
        imids = set([cap["image_id"] for cap in caps])
    except:
        raise Exception(
            "Expect caption file to consist of a dectionary with sentences correspdonding to the key 'imgToEval'"
        )

    return caps, imids, metrics


def save_hallucinated_words(cap_file, cap_dict, output_dir):
    tag = cap_file.split("/")[-1]
    with open(f"{output_dir}_{tag}", "w") as f:
        json.dump(cap_dict, f)


def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict["overall_metrics"]
    metric_string = "%0.01f\t%0.01f\t%0.01f\t%0.01f\t%0.01f" % (
        sentence_metrics["SPICE"] * 100,
        sentence_metrics["METEOR"] * 100,
        sentence_metrics["CIDEr"] * 100,
        sentence_metrics["CHAIRs"] * 100,
        sentence_metrics["CHAIRi"] * 100,
    )

    if not quiet:
        print("SPICE\tMETEOR\tCIDEr\tCHAIRs\tCHAIRi")
        print(metric_string)
        return "SPICE\tMETEOR\tCIDEr\tCHAIRs\tCHAIRi\n" + metric_string
    else:
        return "SPICE\tMETEOR\tCIDEr\tCHAIRs\tCHAIRi\n" + metric_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str, default="")
    parser.add_argument("--annotation_path", type=str, default="coco/annotations")
    args = parser.parse_args()

    _, imids, _ = load_generated_captions(args.cap_file)

    evaluator = CHAIR(imids, args.coco_path)
    evaluator.get_annotations()
    cap_dict = evaluator.compute_chair(args.cap_file)

    print_metrics(cap_dict)
    save_hallucinated_words(args.cap_file, cap_dict)
