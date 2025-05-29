from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from fastapi import Query
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import Counter
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
stop_words = set(stopwords.words("english"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; use specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnswerRequest(BaseModel):
    sentence: str
    word: str
    selected_sense: str

easy_sentences = [
    {"sentence": "As the sun dipped below the horizon, the nocturnal bat, with its leathery wings, silently swooped out of the dimly lit cave, beginning its nightly hunt for insects in the twilight.", "word": "bat"},
    {"sentence": "Carefully, she adorned the beautifully wrapped birthday gift with a delicate, intricately tied pink satin bow, making the present look even more enchanting and inviting.", "word": "bow"},
    {"sentence": "The skilled archer, with unwavering focus, expertly pulled back the taut string of his finely crafted long bow, aiming with precision at the distant bullseye of the target.", "word": "bow"},
    {"sentence": "He meticulously used a black ink pen, its smooth tip gliding effortlessly across the crisp, blank sheet of paper, to elegantly sign his full name on the important legal document.", "word": "pen"},
    {"sentence": "The diligent farmer, after a long day of chores, gently herded the squealing piglets into the sturdy, freshly built wooden pen located at the back of the spacious barn, ensuring their safety for the night.", "word": "pen"},
    {"sentence": "The tranquil bank of the meandering river, softened by years of erosion and covered with a vibrant carpet of lush green grass and wildflowers, provided a serene spot for a peaceful afternoon picnic.", "word": "bank"},
    {"sentence": "Feeling relieved and financially secure, she carefully deposited her hard-earned savings into her new high-interest account at the secure local bank branch, trusting it to keep her money safe for the future.", "word": "bank"},
    {"sentence": "The curious duck, with its bright orange beak, waddled gracefully and settled comfortably on the warm, smooth granite rock that jutted out prominently from the shallow edge of the tranquil pond, enjoying the midday sun.", "word": "rock"},
    {"sentence": "On a sweltering summer afternoon, a group of joyous children splashed and played exuberantly in the cool, crystal-clear water of the sparkling backyard swimming pool, laughing and shouting with delight.", "word": "pool"},
    {"sentence": "He observed with interest how the rough, weathered outer bark of the ancient oak tree was slowly peeling away in large, curling strips, revealing the smoother, lighter wood beneath.", "word": "bark"},
    {"sentence": "Startled by the unexpected sound of a distant ambulance siren, the vigilant guard dog suddenly let out a deep, resonant bark, its loud sound echoing through the quiet evening air, alerting everyone to its presence.", "word": "bark"},
    {"sentence": "The mother hen, clucking contentedly, carefully settled herself in the soft, straw-lined nest, having just laid a perfectly formed, warm brown egg, a new addition to her growing clutch.", "word": "egg"},
    {"sentence": "For their much-anticipated summer vacation, they wisely booked a cozy and well-located hotel room online months in advance, ensuring they had comfortable accommodation for their trip to the coastal town.", "word": "book"},
    {"sentence": "With a contented sigh, she curled up on the comfortable armchair, engrossed in the captivating story of the new fantasy book she had just borrowed from the local library, eagerly turning each page.", "word": "book"},
    {"sentence": "The nimble squirrel, with its bushy tail twitching, began to climb swiftly up the rough trunk of the towering oak tree, easily navigating its way towards a hidden cache of acorns among the branches.", "word": "climb"},
    {"sentence": "After years of rigorous training, the brave mountaineer was finally ready to climb the challenging, snow-capped peak of Mount Everest, a lifelong dream he had diligently pursued.", "word": "climb"},
    {"sentence": "The delicate spider spun a shimmering, intricate web between the branches of the rose bush, patiently waiting for an unsuspecting insect to become entangled in its sticky threads.", "word": "web"},
    {"sentence": "The complex network of interconnected documents on the internet is often referred to as the World Wide Web, a vast resource of information accessible globally.", "word": "web"},
    {"sentence": "As the chef carefully prepared the ingredients, he made sure to peel the thin skin off the fresh potatoes before dicing them for the stew.", "word": "peel"},
    {"sentence": "The banana's bright yellow peel was discarded in the compost bin after he enjoyed the sweet fruit inside.", "word": "peel"},
    {"sentence": "The young artist loved to draw intricate sketches of mythical creatures using her favorite charcoal pencils on large canvases.", "word": "draw"},
    {"sentence": "During the lottery, the winning numbers are drawn randomly from a large machine, creating an air of excitement and anticipation among the participants.", "word": "draw"},
    {"sentence": "He found a shimmering, smooth shell on the beach, perfectly intact and hinting at the mollusk that once inhabited it.", "word": "shell"},
]

intermediate_sentences = [
    {"sentence": "He carefully broke the adhesive seal on the antique letter, a crucial act that allowed him to finally unfold the brittle paper and read the long-awaited, secret message contained within.", "word": "seal"},
    {"sentence": "The playful, sleek-furred seal at the marine park delighted the cheering audience by expertly balancing a colorful ball on its nose and enthusiastically clapping its flippers in response to their applause.", "word": "seal"},
    {"sentence": "The energetic soccer coach blew a sharp, piercing whistle to signal the immediate start of the intense game, causing all the players on the field to spring into action.", "word": "whistle"},
    {"sentence": "As the water reached a rolling boil, the old metal kettle on the stove began to emit a loud, high-pitched whistle, indicating it was ready for making tea, a sound familiar in many kitchens.", "word": "whistle"},
    {"sentence": "With nimble fingers, he swiftly typed a detailed, urgent note to his colleague using the ergonomic keyboard of his sleek laptop, summarizing the important points of their recent meeting.", "word": "note"},
    {"sentence": "The incredibly talented opera singer effortlessly hit a remarkably high note during her powerful vocal performance, thrilling the entire audience and earning a thunderous round of applause.", "word": "note"},
    {"sentence": "The colossal industrial crane, with its long, articulated arm, meticulously lifted an immensely heavy steel beam high into the sky at the bustling construction site, placing it precisely onto the rising framework of the new skyscraper.", "word": "crane"},
    {"sentence": "A majestic, snow-white crane, with its long, slender legs and graceful neck, stood motionless and serene in the tranquil, shallow waters of the marsh, patiently waiting to expertly snatch a fish.", "word": "crane"},
    {"sentence": "Before leaving for the night, she deliberately left a soft, glowing light on in the dimly lit hallway, ensuring that the arriving guests would easily find their way through the unfamiliar house.", "word": "light"},
    {"sentence": "Despite its generous capacity, the innovative new travel suitcase was surprisingly light, allowing her to effortlessly carry it up several flights of stairs without any strain or difficulty.", "word": "light"},
    {"sentence": "During the celebratory breakfast, he enthusiastically raised his glass of freshly squeezed orange juice and made a heartfelt toast to the happy couple, wishing them a lifetime of joy and prosperity.", "word": "toast"},
    {"sentence": "Distracted by the morning news, she accidentally left the slices of bread in the toaster for too long, resulting in the unpleasant smell of distinctly burned toast emanating from the kitchen.", "word": "toast"},
    {"sentence": "The gnarled, weathered bark on the ancient redwood tree was incredibly rough and deeply furrowed to the touch, telling tales of centuries of exposure to the elements.", "word": "bark"},
    {"sentence": "The alert guard dog, sensing an unfamiliar presence approaching the property, let out a deep, resonating bark that echoed loudly through the otherwise quiet night, serving as a clear warning.", "word": "bark"},
    {"sentence": "The fluffy, yellow baby duck, chirping happily, waddled adorably and dutifully followed its mother in a neat line as she gracefully led her brood into the cool, shimmering waters of the serene pond.", "word": "duck"},
    {"sentence": "Reacting instantly and with remarkable agility, he had to quickly duck his head just as the fast-approaching baseball, thrown with immense force, whizzed narrowly past where his head had been.", "word": "duck"},
    {"sentence": "The extraordinarily bright star, Venus, could be observed twinkling vividly and distinctly in the clear, unpolluted night sky, standing out prominently among the other celestial bodies.", "word": "star"},
    {"sentence": "After her critically acclaimed performance in the blockbuster film, she rapidly rose to international fame and became an undeniable movie star, recognized and adored by millions worldwide.", "word": "star"},
    {"sentence": "Using a small, sharp hammer, he carefully tapped the slender metal nail into the drywall, intending to securely hang the newly framed painting on the freshly painted living room wall.", "word": "nail"},
    {"sentence": "He struck the tiny wooden match against the rough surface of the matchbox, producing a small flame that he then used to successfully ignite the kindling for the roaring campfire.", "word": "match"},
    {"sentence": "The two rival teams played an incredibly intense and closely contested soccer match on Saturday afternoon, with both sides exhibiting remarkable skill and determination until the very last minute.", "word": "match"},
    {"sentence": "He positioned himself directly in front of the powerful oscillating fan, enjoying the refreshing breeze that it generated, which provided a much-needed respite from the oppressive summer heat.", "word": "fan"},
    {"sentence": "She revealed herself to be an extremely passionate and loyal fan of that particular indie music group, having attended every single one of their concerts in the city for the past five years.", "word": "fan"},
    {"sentence": "The remarkable parrot, with its vibrant green and blue feathers, astonished visitors by its ability to clearly speak several complex words and phrases using its agile and surprisingly human-like tongue.", "word": "tongue"},
    {"sentence": "The small, flat sole of his worn-out running shoe had a hole, making it uncomfortable to walk on uneven surfaces during his long morning run.", "word": "sole"}
]

difficult_sentences = [
    {"sentence": "After careful deliberation of all the evidence presented, the stern judge delivered a harsh ten-year prison sentence to the convicted thief, a punishment intended to deter future criminal acts.", "word": "sentence"},
    {"sentence": "The diligent English teacher instructed her elementary school class to construct a grammatically correct and coherent complete sentence, emphasizing the importance of a subject and a predicate.", "word": "sentence"},
    {"sentence": "The deeply moving fable concluded with a profound moral about the enduring importance of kindness and empathy towards all living creatures, a timeless lesson for young and old alike.", "word": "moral"},
    {"sentence": "Despite facing immense pressure and the tempting offer of illicit gains, the principled soldier steadfastly maintained his unwavering moral compass, refusing to compromise his integrity or principles.", "word": "moral"},
    {"sentence": "The charismatic politician is currently actively campaigning and diligently running for a significant public office in the highly anticipated upcoming national election, hoping to represent his constituents.", "word": "office"},
    {"sentence": "She arrived promptly at her professional office building precisely at nine in the morning, ready to diligently begin her workday at her meticulously organized desk, preparing for a day full of meetings.", "word": "office"},
    {"sentence": "The critically acclaimed novel intricately explores the complex and universal theme of profound loss and subsequent emotional recovery, delving deeply into the human experience of grief and healing.", "word": "theme"},
    {"sentence": "Politely, she asked, 'Do you mind if I politely sit next to you on this crowded park bench? I promise to be quiet and not disturb your reading,' hoping for his kind permission.", "word": "mind"},
    {"sentence": "They meticulously used the intricate, complex decryption code to successfully unlock the seemingly indecipherable hidden message, revealing a crucial piece of information for their clandestine mission.", "word": "code"},
    {"sentence": "The talented software engineer diligently spent countless hours writing lines of intricate computer code to painstakingly build the innovative and user-friendly new e-commerce website from scratch.", "word": "code"},
    {"sentence": "During the powerful storm, the fierce wind created an eerie, mournful whistle as it relentlessly swept through the dense branches of the ancient trees, rattling the windowpanes.", "word": "whistle"},
    {"sentence": "The strict football referee, with a decisive gesture, blew his shrill whistle loudly to abruptly halt the heated game, signaling an immediate foul and stopping the play.", "word": "whistle"},
    {"sentence": "Feeling fatigued after a long shift, she gratefully left her designated work post to take a much-needed, short break, stepping away from her responsibilities for a few minutes.", "word": "post"},
    {"sentence": "He enthusiastically made a detailed and highly informative post on his personal social media account about the upcoming charity event, aiming to garner significant interest and widespread participation from his followers.", "word": "post"},
    {"sentence": "Based on the comprehensive and meticulously gathered data, the experienced scientist was able to draw a very accurate and statistically sound conclusion, contributing significantly to the research.", "word": "draw"},
    {"sentence": "The young children in the art class absolutely love to draw colorful and imaginative pictures of fantastical creatures and vibrant landscapes using their brand new boxes of crayons.", "word": "draw"},
    {"sentence": "She is an exceptionally bright and intellectually curious student who grasps complex concepts with remarkable speed and an impressive depth of understanding, consistently excelling in all her academic subjects.", "word": "bright"},
    {"sentence": "The newly installed LED lamp in the living room is incredibly bright, illuminating the entire space with a brilliant, clear light, making it perfect for reading and intricate tasks.", "word": "bright"},
    {"sentence": "She promptly filed a formal insurance claim for the valuable antique vase that had been unfortunately broken during transit, providing all necessary documentation for reimbursement.", "word": "claim"},
    {"sentence": "The eccentric man made a bold and astonishing claim that he had personally witnessed a genuine UFO hovering silently above his backyard just last night, a statement met with skepticism.", "word": "claim"},
    {"sentence": "The accomplished musician decided to score the entire soundtrack for the new independent film, composing original music that perfectly captured the mood and narrative of the movie.", "word": "score"},
    {"sentence": "He achieved a perfect score of 100% on the incredibly challenging mathematics exam, demonstrating his exceptional understanding of calculus and proving his academic prowess.", "word": "score"},
    {"sentence": "The experienced police officer was carefully trained to bear the weight of responsibility that came with his badge and uniform, always upholding the law with integrity and courage.", "word": "bear"}
]

def clean_tokens(tokens):
    return [
        word.lower() for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None

def get_best_sense_sbert(sentence: str, ambiguous_word: str):
    synsets = wn.synsets(ambiguous_word)
    if not synsets:
        return None

    tokens = word_tokenize(sentence)
    context = " ".join([w for w in tokens if w.lower() != ambiguous_word.lower()])
    context_embedding = model.encode(context, convert_to_tensor=True)

    best_score = -1.0
    best_sense = None
    for synset in synsets:
        def_embedding = model.encode(synset.definition(), convert_to_tensor=True)
        score = util.cos_sim(context_embedding, def_embedding).item()
        if score > best_score:
            best_score = score
            best_sense = synset

    return best_sense

def generate_choices_sbert(sentence: str, ambiguous_word: str, num_choices: int = 4):
    correct_sense = get_best_sense_sbert(sentence, ambiguous_word)
    if not correct_sense:
        return [], None

    all_senses = wn.synsets(ambiguous_word)
    if len(all_senses) <= num_choices:
        distractors = [s for s in all_senses if s != correct_sense]
    else:
        tokens = word_tokenize(sentence)
        context = " ".join([w for w in tokens if w.lower() != ambiguous_word.lower()])
        context_embedding = model.encode(context, convert_to_tensor=True)

        scored_distractors = []
        for syn in all_senses:
            if syn != correct_sense:
                def_embedding = model.encode(syn.definition(), convert_to_tensor=True)
                score = util.cos_sim(context_embedding, def_embedding).item()
                scored_distractors.append((syn, score))

        distractors = [
            s for s, _ in sorted(scored_distractors, key=lambda x: x[1], reverse=True)
        ][:num_choices - 1]

    # Combine and shuffle
    final_choices = [correct_sense] + distractors
    random.shuffle(final_choices)

    return [s.definition() for s in final_choices], correct_sense.definition()

@app.get("/get-question")
def get_question(difficulty: str = Query("easy", enum=["easy", "intermediate", "difficult"])):
    if difficulty == "easy":
        sample = random.choice(easy_sentences)
    elif difficulty == "intermediate":
        sample = random.choice(intermediate_sentences)
    else:
        sample = random.choice(difficult_sentences)

    sentence = sample["sentence"]
    word = sample["word"]
    choices, _ = generate_choices_sbert(sentence, word)

    return {
        "sentence": sentence,
        "word": word,
        "choices": choices
    }


@app.post("/submit-answer")
def check_answer(req: AnswerRequest):
    word = req.word
    sentence = req.sentence
    selected = req.selected_sense

    synsets = wn.synsets(word)

    if not synsets:
        return {"correct": False, "correct_sense": None}

    tokens = word_tokenize(sentence)
    context = " ".join([w for w in tokens if w.lower() != word.lower()])

    context_embedding = model.encode(context, convert_to_tensor=True)

    best_sense = get_best_sense_sbert(sentence, req.word)

    is_correct = best_sense.definition() == selected

    return {
        "correct": is_correct,
        "correct_sense": best_sense.definition()
    }