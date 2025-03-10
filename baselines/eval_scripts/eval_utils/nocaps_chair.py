
import pandas as pd
import json
import copy
import numpy as np

import re
import nltk
from nltk.corpus import brown

# # Download necessary NLTK data
# nltk.download('wordnet')
# nltk.download('brown')

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



def convert_to_readable_hierachy():
    hierachy_path = './data_nocaps/bbox_labels_600_hierarchy.json'
    # label_names_csv = './oidv7-class-descriptions-boxable.csv'
    label_names_csv = './data_nocaps/oidv7-class-descriptions.csv'
    label_names_pd = pd.read_csv(label_names_csv, header=None)

    hierachy_info = json.load(open(hierachy_path, 'r'))
    new_json = copy.deepcopy(hierachy_info)

    # json to str
    json_str = json.dumps(hierachy_info)

    label_maps = {}
    for item in label_names_pd.values:
        label_maps[item[0]] = item[1]

    for k, v in label_maps.items():
        json_str = json_str.replace(k, v)

    # save json_str to new json_file in indented format
    with open('./openimages_hierachy.json', 'w') as f:
        f.write(json.dumps(json.loads(json_str), indent=4))
        

def get_common_synonyms(word, min_frequency=20):

    # Create a frequency distribution of words in the Brown corpus
    freq_dist = nltk.FreqDist(w.lower() for w in brown.words())

    synonyms = set()
    for syn in wordnet.synsets(word, 'n'):
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace('_', ' ').lower()
            # Check the frequency of the lemma in the corpus
            if freq_dist[lemma_name] >= min_frequency:
                synonyms.add(lemma_name)
    return list(synonyms)

from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word, 'n'):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

from nltk.corpus import wordnet as wn

def is_category_member(word, category):
    synsets = wn.synsets(word)
    for synset in synsets:
        # Check all hypernym paths for each synset
        for hypernym_path in synset.hypernym_paths():
            for hypernym in hypernym_path:
                if category in hypernym.name():
                    return True
    return False


def get_corse_class_labels():
    hierachy_path = './openimages_hierachy.json'
    hierachy_info = json.load(open(hierachy_path, 'r'))
    hierachy_info = hierachy_info['Subcategory']
    super_classes = []
    for item in hierachy_info:
        super_classes.append(item['DisplayName'])

    return super_classes

def get_mid_class_labels():
    hierachy_path = './openimages_hierachy.json'
    hierachy_info = json.load(open(hierachy_path, 'r'))
    hierachy_info = hierachy_info['Subcategory']
    super_classes = []
    for item in hierachy_info:
        if 'Subcategory' in item.keys():
            for sub_item in item['Subcategory']:
                super_classes.append(sub_item['Subcategory'])
        else:
            super_classes.append(item['Subcategory'])

    return super_classes


def get_fine_class_labels(args):
    from data.nocaps import NoCapsDataset
    nocaps_ds = NoCapsDataset(args['data_path'], 'out-domain')
    # all_classes = list(set([c.lower() for c in nocaps_ds.all_classes]))
    all_classes = list(set([c for c in nocaps_ds.all_classes]))

    class_maps = {}
    for c in all_classes:
        class_maps[c] = get_synonyms(c)
    print('len class_maps: {}'.format(len(class_maps)))
    print('class_maps: {}'.format(class_maps))


def extract_subcategories(node, superclasses=[]):
    label = node['DisplayName']
    children = node.get('Subcategory', [])

    subcategories = {label: []} if label.lower() in superclasses else {}

    for child in children:
        child_label = child['DisplayName']
        if label.lower() in superclasses:
            subcategories[label].append(child_label)
        child_subcategories = extract_subcategories(child, superclasses)
        for superclass, subclasses in child_subcategories.items():
            if superclass in subcategories:
                subcategories[superclass].extend(subclasses)
            else:
                subcategories[superclass] = subclasses

    return subcategories

# class Node:
#     def __init__(self, label_name, parents) -> None:
#         self.label_name = label_name

def build_tree(node):
    label = node['DisplayName']
    children = node.get('Subcategory', [])
    
    
    tree = {label: [label]}
    
    for child in children:
        tree[label].append(child['DisplayName'])
        
        child_tree = build_tree(child)
        
        tree.update(child_tree)
        tree[label].extend(child_tree[child['DisplayName']])
    
    if "Part" in node:
        part_nodes = node["Part"]
        for part_node in part_nodes:
            tree.update(build_tree(part_node))
            
    return tree

from tqdm import tqdm
def extract_subcate_for_superclasses(superclasses, hierachy_info, add_synonyms=True, is_lower=True):
    tree = build_tree(hierachy_info)
    print(len(tree.keys()))
    print(len(superclasses))
    out = {}
    for superclass in superclasses:
        if superclass not in tree.keys():
            print(f"Warning: {superclass} not in tree")
        out[superclass] = tree[superclass]
    
    # remove duplicates
    out = {k: list(set(v)) for k, v in out.items()}
    _out = copy.deepcopy(out)
    if add_synonyms:
        for k, v in tqdm(_out.items()):
            for item in v:
                out[k].extend(get_common_synonyms(item))
    # lower
    if is_lower:
        out = {k.lower(): [item.lower() for item in v] for k, v in out.items()}
    out = {k: list(set(v)) for k, v in out.items()}
    
    return out

def collect_groundtruth_labels(args):
    from nocaps import NoCapsDataset
    
    caption_path = '{}/nocaps_val_4500_captions.json'.format(args['data_path'])
    ds = NoCapsDataset(args['data_path'], args['domain'])
    
    caption_json = json.load(open(caption_path, 'r'))
    caption_info = caption_json['images']
    caption_annot = caption_json['annotations']
    
    out = []
    for i, sample in tqdm(enumerate(ds)):
        image_id = ds.get_image_id(i)
        
        # find the corresponding image info
        nocaps_id = [item['id'] for item in caption_info if item['open_images_id'] == image_id][0]
        captions = [item['caption'] for item in caption_annot if item['image_id'] == nocaps_id]
        
        out.append({
            'image_id': ds.get_image_id(i),
            'labels': sample[1],
            'human_captions': captions
        })
    
    return out

def collect_groundtruth_labels_w_seem(args, seem_path=''):
    from nocaps import NoCapsDataset
    
    caption_path = '{}/nocaps_val_4500_captions.json'.format(args['data_path'])
    ds = NoCapsDataset(args['data_path'], args['domain'])
    
    caption_json = json.load(open(caption_path, 'r'))
    caption_info = caption_json['images']
    caption_annot = caption_json['annotations']
    
    # seem_list = json.loads(open(seem_path, 'r').read())
    # seem_list = json.loads(open(seem_path, 'r').read())
    seem_list =  [json.loads(q) for q in open(seem_path, 'r')]
    seem_map = {}
    for item in seem_list:
        # only get image id from image path
        image_id = item['image'].split('/')[-1].split('.')[0]
        # print(image_id)
        seem_map[image_id] = item['objects']
    
    out = []
    for i, sample in tqdm(enumerate(ds)):
        image_id = ds.get_image_id(i)
        
        # find the corresponding image info
        nocaps_id = [item['id'] for item in caption_info if item['open_images_id'] == image_id][0]
        captions = [item['caption'] for item in caption_annot if item['image_id'] == nocaps_id]
        
        seem_res = seem_map[image_id]
        
        out.append({
            'image_id': ds.get_image_id(i),
            'labels': sample[1],
            'human_captions': captions,
            'seem_labels': seem_res
        })
    
    return out
        


import nltk
import os
cur_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_nocaps')

class CHAIR(object):
    
    def __init__(self, image_ids, args, verbose=False):

        self.verbose = verbose
        self.image_ids = image_ids

        
        self.synothms = json.load(open(os.path.join(cur_path, 'nocaps_corse_classlabels_lower.json'), 'r'))
        
        # inverse the synothms
        self.synothms_inv = {}
        for k, v in self.synothms.items():
            for item in v:
                self.synothms_inv[item] = k

        
        self.labels = json.load(open(os.path.join(cur_path, 'nocaps_corse_classlabels.json'), 'r'))
        self.labels_inv = {}
        for k, v in self.labels.items():
            for item in v:

                self.labels_inv[item] = k   
        
        # collect ground truth labels
        # if args.get('eval_seem_labels', False):
        #     self.gt_info = json.load(open(os.path.join(cur_path, 'nocaps_{}_collected_gt_w_seem.json'.format(args['domain'])), 'r'))
        # else:
        self.gt_info = json.load(open(os.path.join(cur_path, 'nocaps_{}_collected_gt.json'.format(args.domain)), 'r'))

        self.imid_to_objects = {imid: [] for imid in self.image_ids}
        self.verbose_imid_to_objects = {imid: [] for imid in self.image_ids}
        
        self.get_annotations_from_labels()
        print('loaded annotations from labels')
        self.get_annotations_from_captions()
        print('loaded annotations from captions')

        if self.gt_info[0].get('seem_labels', None) is not None:
            self.get_annotations_from_seem()
            print('loaded annotations from seem labels')
        
        
        self.clean_annotations()
    
    def clean_annotations(self):
        for imid, objects in self.imid_to_objects.items():
            # lower
            self.imid_to_objects[imid] = list(set([o.lower() for o in objects]))
            # self.imid_to_objects[imid] = list(set(objects))
    
    def get_double_words_dict(self, all_available_words):
        double_word_dict = {}
        for word in all_available_words:
            if len(word.split(' ')) == 2:
                double_word_dict[' '.join(word.split(' ')[:2])] = word
        return double_word_dict
        
        
    def get_annotations_from_labels(self):
        for item in self.gt_info:
            if item['image_id'] in self.image_ids:
                for label in item['labels']:
                    if self.verbose:
                        print('label word', label, self.labels_inv[label])
                    self.imid_to_objects[item['image_id']].append(self.labels_inv[label])
    
    def get_annotations_from_captions(self):
        for item in self.gt_info:
            if item['image_id'] in self.image_ids:
                for caption in item['human_captions']:
                    words, node_words, idxs, double_words = self.caption_to_words(caption)
                    for idx, word in enumerate(node_words):
                        if word in self.synothms_inv.keys():
                            self.imid_to_objects[item['image_id']].append(self.synothms_inv[word])
                    # self.verbose_imid_to_objects[item['image_id']].append()
    
    def get_annotations_from_seem(self):
        for item in self.gt_info:
            if item['image_id'] in self.image_ids:
                for caption in item['seem_labels']:
                    words, node_words, idxs, double_words = self.caption_to_words(caption)
                    for idx, word in enumerate(node_words):
                        if word in self.synothms_inv.keys():
                            self.imid_to_objects[item['image_id']].append(self.synothms_inv[word])

    
    
    def caption_to_words(self, caption):
        
        all_available_words = self.synothms_inv.keys()
        # get dowble words dict in all available words
        self.double_word_dict = self.get_double_words_dict(all_available_words)
        
        
        #standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        words = [singularize(w) for w in words]
        
        
        #replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
           idxs.append(i) 
           double_word = ' '.join(words[i:i+2])
           if double_word in self.double_word_dict: 
               double_words.append(self.double_word_dict[double_word])
               i += 2
           else:
               double_words.append(words[i])
               i += 1
        words = double_words
        
        #get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(all_available_words)]
        words = [word for word in words if word in set(all_available_words)]
        node_words = []
        for word in words:
            if self.verbose:
                print('caption word', word, self.synothms_inv[word])
            node_words.append(self.synothms_inv[word])
            
        return words, node_words, idxs, double_words

    def compute_chair(self, caps):
        # caps, imids, _ = self.load_cap_file(cap_file)
        self.caps = caps
        
        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.

        output = {'sentences': []} 
        
        for i, cap_eval in enumerate(caps):
            cap = cap_eval['caption']
            imid = cap_eval['image_id']
            
            gt_objects = self.imid_to_objects[imid]
                
            #get all words in the caption, as well as corresponding node word
            words, node_words, idxs, raw_words = self.caption_to_words(cap) 
 
            cap_dict = {'image_id': cap_eval['image_id'], 
                        'caption': cap,
                        'valid_hallucinated_words': [],
                        'valid_gt_words': list(gt_objects),
                        'valid_generated_words': list(node_words),
                        'hallucination_idxs': [], 
                        'valid_gt_words_detail': [],
                        'valid_generated_words_detail': [],
                        # 'words': raw_words,
                        }
            cap_dict['metrics'] = {
                                   'CHAIRs': 0,
                                   'CHAIRi': 0}
 
            #count hallucinated words
            coco_word_count += len(node_words) 
            hallucinated = False
            for word, node_word, idx in zip(words, node_words, idxs):
                cap_dict['valid_generated_words_detail'].append((word, node_word, idx))
                if node_word not in gt_objects:
                    hallucinated_word_count += 1 
                    cap_dict['valid_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True
                else:
                    cap_dict['valid_gt_words_detail'].append((word, node_word, idx))
    
            #count hallucinated caps
            num_caps += 1
            if hallucinated:
               num_hallucinated_caps += 1
    
            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.
            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['valid_hallucinated_words'])/float(len(words))
   
            output['sentences'].append(cap_dict)
 
        chair_s = (num_hallucinated_caps/num_caps)
        chair_i = (hallucinated_word_count/coco_word_count)
        
        print('considered caption number: %d' %num_caps)
    
        output['overall_metrics'] = {
                                     'CHAIRs': chair_s,
                                     'CHAIRi': chair_i}
    
        return output 
        
    
    def load_cap_file(self, cap_file):
        caps = json.load(open(cap_file))
    
        try:
            items = []
            imids = []
            for c in caps:
                items.append({
                    'image_id': c['image_id'],
                    'caption': c['answer'],
                })
                imids.append(c['image_id'])
            imids = set(imids)
        except:
            raise Exception("Caption file should be a list of dictionaries with keys 'image_id' and 'answer'")
        
        return items, imids, None

if __name__ == '__main__':

    # --------------- collect gt information ----------------
    args = {
        'data_path': './dataset/nocaps/annotations/',
        'domain': 'out-domain',
        'answers_file': 'captions.json'
    }

    # --------------- compute CHAIR ----------------

    answers = []
    for line in open(args['answers_file']):
        ans = json.loads(line)
        answer = {
             "caption": ans['text'],
             "image_id": os.path.splitext(ans['image'])[0],
             "image": ans['image'],
        }
        answers.append(answer)

    imids = [answer['image_id'] for answer in answers]

    evaluator = CHAIR(imids, args, verbose=True)
    cap_dict = evaluator.compute_chair(answers)

    # save to json pretty print
    chair_json_path = args['answers_file'].replace('.json', '_eval_results.json')
    assert chair_json_path != args['answers_file']
    
    with open(chair_json_path, "w") as f:
        json.dump(cap_dict, f, indent=4)

    print(cap_dict['overall_metrics'])
