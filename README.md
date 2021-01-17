# Discourse-Structure-Classifier
Mass article discourse structure tagger developed for Wharton, based on work and code from https://www.aclweb.org/anthology/2020.acl-main.478.pdf

# Installation

Run in Terminal:

pip3 install torch  
pip3 install allennlp  
pip3 install nltk  
pip3 install pandas  

Then, inside Python:  

import nltk 
nltk.download('punkt')

# Usage

Article texts go in tester.csv. Then, run generate_tags.py.

# Tags

## Main Contents

Main Event (M1) introduces the most important event and relates to the major subjects in a
news report. It follows strict constraints of being the most recent and relevant event, and directly
monitors the processing of remaining document. Categories of all other sentences in the
document are interpreted with respect to the main event.

Consequence (M2) informs about the events that are triggered by the main news event. They
are either temporally overlapped with the main event or happens immediately after the main
event.

## Context-informing Contents
Context-informing sentences provide information related to the actual situation in which main
event occurred. It includes the previous events and other contextual facts that directly explain
the circumstances that led to the main event.

Previous Event (C1) describes the real events that preceded the main event and now act as
possible causes or preconditions for the main event. They are restricted to events that have
occurred very recently, within last few weeks.

Current Context (C2) covers all the information that provides context for the main event. They
are mainly used to activate the situation model of current events and states that help to
understand the main event in the current social or political construct. They have temporal
co-occurrence with the main event or describe the ongoing situation.

## Additional Supportive Contents

Finally, sentences containing the least relevant information, comprising of unverifiable or
hypothetical facts, opinionated statements, future projections and historical backgrounds, are
classified as distantly-related content.

Historical Event (D1) temporally precedes the main event in months or years. It constitutes the
past events that may have led to the current situation, or indirectly relates to the main event or
subjects of the news article.

Anecdotal Event (D2) includes events with specific participants that are difficult to verify. It may
include fictional situations or personal account of incidents of an unknown person especially
aimed to exaggerate the situation.

Evaluation (D3) introduces reactions from immediate participants, experts or known
personalities that are opinionated and may also include explicit opinions of the author or those
of the news source. They are often meant to describe the social or political implications of the
main event or evaluation of the current situation. Typically, it uses statements from influential
people to selectively emphasize on their viewpoints.

Expectation (D4) speculates on the possible consequences of the main or contextual events.
They are essentially opinions, but with far stronger implications where the author tries to
evaluate the current situation by projecting possible future events.

## Misc
NA -- Sentences that do not contribute to the discourse structure such as photo captions, text links for
images, etc.
