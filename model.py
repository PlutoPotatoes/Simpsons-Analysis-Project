import csv
import pandas as pd
from pandas import DataFrame
import numpy as np
from collections import defaultdict

'''
Get average rating across all episodes
Base Model: Impact of characters on episode rating diff (weight by number of lines for each character)
  - get average episode rating when character is present
  - weight rating contribution by percentage of lines in the episode

 Adjustment 1: (rating diff from average)/views to get a ratio, test how close it is to actual rating (final fix, shift after other models)
 Adjustment 2: Season average rating
 Adjustment 4: 

Total average = 7.3491...

'''


episodes = "simpsons_episodes.csv"
lines = "simpsons_script_lines.csv"

epdf = pd.read_csv(episodes, encoding='utf-8', encoding_errors='ignore')
linedf = pd.read_csv(lines, encoding = 'utf-8', encoding_errors='ignore', low_memory=False)


'''
episodeList = []
linedf = []

with open(episodes, 'r') as f:
    dict_reader = csv.DictReader(f)
    episodeList = list(dict_reader)

print(episodeList[1].keys())
avgRating = 0
for episode in episodeList:
    if episode['imdb_rating'] != '':
        avgRating += float(episode['imdb_rating'])

avgRating = avgRating/len(episodeList)
print(avgRating)

for episode in episodeList:
    #get sums of character ids with lines per episode
    #build/normalized episode profile
    # rating * percentage of lines for each episode = episode impact score

    id = episode['id']
'''


epLines = linedf[linedf['episode_id'] == 32]
'''
charScores = []
for charID in linedf['character_id'].unique()[:5]:
    print(charID)
    scores = []
    for epID in linedf['episode_id'].unique():
        totalLines = len(linedf[linedf['episode_id'] == epID])
        charLines = len(linedf[(linedf['episode_id'] == epID) & (linedf['character_id'] == charID)])
        epScore = epdf[epdf['id'] == epID]['imdb_rating']

        #calculate score contributed per line by the character
        perLineScore = (epScore * (charLines/totalLines))/epScore
        scores.append(perLineScore)
    charScores.append(tuple([charID, np.mean(scores)]))
'''


char_scores = defaultdict(list)
for epID in linedf['episode_id'].unique()[:300]:
    epScore = epdf[epdf['id'] == epID]['imdb_rating']
    lines = linedf[linedf['episode_id'] == epID]
    for charId in lines['character_id'].unique():
        if charId is not None:
            line_count = lines[lines['character_id'] == charId]['character_id'].count()
            line_percentage = line_count/len(lines)
            score_percentage = line_percentage*epScore
            char_scores[charId].append(float(score_percentage.iloc[0]))


#print(char_scores)

epId = linedf['episode_id'].head(1).reset_index(drop=True)
epId = 255
lines = linedf[linedf['episode_id'] == epId]

totalScore = 0
for charId in lines['character_id'].unique():
    line_count = lines[lines['character_id'] == charId]['character_id'].count()
    line_percentage = line_count/len(lines)
    score = line_percentage * np.mean(char_scores[charId])
    totalScore += score

print(totalScore*10)
