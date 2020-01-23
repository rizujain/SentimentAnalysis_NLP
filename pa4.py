import sys
import getopt
import os
import math
import operator
import re

from collections import defaultdict

USE_MORE = False
class SemanticOrientation:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []
      self.phrases = []


  def __init__(self):
    """SemanticOrientation initialization"""
    self.numFolds = 10

    self.near_excellent = defaultdict(int)
    self.near_poor = defaultdict(int)
    self.hits_excellent = 0
    self.hits_poor = 0


  def getPhrases(self,abs_fileName):

    first_regex= "(\w*'*\w*_JJ\w*-\w*?)\s(\w*_NN[S]?\w*-*\w+)\s(\w*)"
    second_regex = "(\w*'*\w*_RB[RS]?_\w*-\w*?)\s(\w*_JJ\w*-*\w+)\s(\w+_(?!NN[S]?).\w+-\w*)"
    third_regex = "(\w*'*\w*_JJ_\w*-\w*?)\s(\w*_JJ\w*-*\w+)\s(\w+_(?!NN[S]?).\w+-\w*)"
    fourth_regex = "(\w*'*\w*_NN[S]?_\w*-\w*?)\s(\w*_JJ\w*-*\w+)\s(\w+_(?!NN[S]?).\w+-\w*)"
    fifth_regex = "(\w*'*\w*_RB[RS]?\w*-\w*?)\s(\w*_VB[DNG]?\w*-*\w+)\s(\w*)"
    f = open(abs_fileName)

    matches = []
    text = f.read()
    matches.extend(re.findall(first_regex,text))
    matches.extend(re.findall(second_regex,text))
    matches.extend(re.findall(third_regex,text))
    matches.extend(re.findall(fourth_regex,text))
    matches.extend(re.findall(fifth_regex,text))

    phrases = []
    for match in matches:
      words1 = match[0].split("_")
      words2 = match[1].split("_")

      phrases.append((words1[0],words2[0]))

    return phrases

  def classify(self, phrases):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    avg_SO = 0

    for phrase in phrases:
      phrase_hits_excellent = self.near_excellent[phrase]
      phrase_hits_poor = self.near_poor[phrase]

      if phrase_hits_poor < 1 and phrase_hits_excellent < 1:
        # skip to the next phrase as mentioned in the paper
        continue

      phrase_hits_excellent += 0.01
      phrase_hits_poor += 0.01

      numerator = phrase_hits_excellent * self.hits_poor
      denominator = phrase_hits_poor * self.hits_excellent

      phrase_SO = math.log(numerator/denominator)
      avg_SO += phrase_SO

    ans = ('neg','pos')[avg_SO>=0]
    return ans


  def addExample(self, klass, words):
    global USE_MORE
    # when we are done training we will have the exact hit count for each of the words
    # and dictionaries that maintains the counts for neighbor phrases
    just_words = [new_word.split("_")[0] for new_word in words]

    if not USE_MORE:
      pos_words = ("excellent")
      neg_words = ("poor")
    else:
      pos_words = ("excellent","good","best")
      neg_words = ("poor","bad","worst")

    for idx,word in enumerate(just_words):
      if word in pos_words:
        self.hits_excellent += 1
        min_id = max(0,idx-10)
        max_id = min(idx+10+1,len(just_words))

        for neighbor_id in range(min_id,max_id-1):
          self.near_excellent[(just_words[neighbor_id],just_words[neighbor_id+1])] += 1

      elif word in neg_words:
        self.hits_poor += 1
        min_id = max(0,idx-10)
        max_id = min(idx+10+1,len(just_words))

        for neighbor_id in range(min_id,max_id-1):
          self.near_poor[(just_words[neighbor_id],just_words[neighbor_id+1])] += 1



  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here,
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents))
    return result


  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()


  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
    """Returns a list of TrainSplits corresponding to the cross validation splits."""
    splits = []
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          abs_fname = '%s/pos/%s' % (trainDir, fileName)
          example.phrases = self.getPhrases(abs_fname)
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          abs_fname = '%s/neg/%s' % (trainDir, fileName)
          example.phrases = self.getPhrases(abs_fname)
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits

  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(dirName):
  so = SemanticOrientation()
  splits = so.crossValidationSplits(dirName)
  print("cv done")
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = SemanticOrientation()
    accuracy = 0.0
    for example in split.train:
      words = example.words
      classifier.addExample(example.klass, words)

    for example in split.test:
      phrases = example.phrases
      guess = classifier.classify(phrases)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print( '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) )
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print ('[INFO]\tAccuracy: %f' % avgAccuracy)


def main():
  global USE_MORE

  if len(sys.argv) > 2 and sys.argv[2] == "-m":
    USE_MORE = True
  
  test10Fold(sys.argv[1])


if __name__ == "__main__":
    main()
