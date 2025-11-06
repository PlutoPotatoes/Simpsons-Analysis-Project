import csv
import pandas as pd
from pandas import DataFrame

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

linedf = pd.read_csv(lines, encoding = 'utf-8', encoding_errors='ignore')

for episode in episodeList:
    #get sums of character ids with lines per episode
    #build/normalized episode profile
    # rating * percentage of lines for each episode = episode impact score

    id = episode['id']



epLines = linedf[linedf['episode_id'] == 32]
for charID in epLines['character_id'].unique():
    print(charID)
