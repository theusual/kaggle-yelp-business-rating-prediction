import sys
import json
import csv
import time


def loadSentimentData(filePath):
	termScores = {} # initialize an empty dictionary
	counter = 0
	for line in filePath:
		term, score  = line.split("\t")  # The file is tab-delimited.
		termScores[term] = int(score)  # Convert the score to an integer.
	return termScores

def getJSONData(filePath):
	jsonData = {}
	counter = 1

	for line in filePath:
		jsonLine = json.loads(line)
		jsonLineKey = jsonLine.get('review_id')
		jsonData[jsonLineKey] = jsonLine
		counter += 1
		#print type(counter) , type(readLineMax)
		#if counter > int(readLineMax):
			#return jsonData
	return jsonData

def calcReviewSentimentScore(termScores,jsonReviews):
	scoredReviews = {}
	counter = 0

	for reviewID in jsonReviews:
		words = jsonReviews[reviewID].get('text').split()
		sentimentScore = 0
		scoredReviews[reviewID] = (0,0)
		for word in words:
			if word in termScores.keys():
				sentimentScore += termScores[word]
		scoredReviews[reviewID] = (sentimentScore, reviewID)
		counter += 1
		#print 'Reviews processed:', counter
	return scoredReviews

def writeCSV(dataDict):
	with open('Data/yelp_training_set_sent_score.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile)
		for key in dataDict:
			writer.writerow(dataDict[key])

def findMin(field,jsonData):
	fieldList = []
	for key in jsonData:
		fieldList.append(jsonData[key].get(field))
	return min(fieldList)

def findMax(field,jsonData):
	fieldList = []
	for key in jsonData:
		fieldList.append(jsonData[key].get(field))
	return max(fieldList)
		

def main():
    start_time = time.time()
    print time.asctime( time.localtime(time.time()) ) + ' ---  Program started'
    json_file = open("Data/yelp_training_set_review.json")
    jsonData = getJSONData(json_file)
    sent_file = open("AFINN-111.txt")
    termScores = loadSentimentData(sent_file)
    print time.asctime( time.localtime(time.time()) ) + ' ---  Calculating sentiment scores...'
    scoredReviews = calcReviewSentimentScore(termScores,jsonData)
    #print scoredReviews
    print time.asctime( time.localtime(time.time()) ) + ' ---  Writing csv output...'
    writeCSV(scoredReviews)
    execution_time = time.time() - start_time
    print time.asctime( time.localtime(time.time()) ) + ' ---  Program completed.  Execution time (mins): ' + str(execution_time / 60)

if __name__ == '__main__':
    main()
