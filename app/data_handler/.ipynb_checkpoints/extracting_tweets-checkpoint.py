#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import json
import os
import sys
from app.path_manager import Datapath

#os.chdir('/users/tweetcontextualization/thoang/Collections/SIGIR/SecondWeek')
def Get_text_from_tweet_json(file_in_json, file_out_text):
    f_in = open(file_in_json, "r")
    f_out_text = open(file_out_text, "w")
    for line in f_in:
        tweet = json.loads(line)
        Tweet_content = tweet['text'].encode('utf-8')
        Tweet_content = Tweet_content.replace('\n', '').replace('\r', '')
        f_out_text.write("{0}\n".format(Tweet_content))
    f_in.close()
    f_out_text.close()


def get_first_OriginalTweet(file_jason_in, file_info_out, file_id_orgininal_tweet_out,file_orgininal_tweet_out,number_original_tweet_wanted):
    f_in = open(file_jason_in,'r')
    f_info_out= open(file_info_out,'w')
    f_id_orgininal_tweet_out = open(file_id_orgininal_tweet_out, 'w')
    f_orgininal_tweet_out = open(file_orgininal_tweet_out, 'w')
    id_tweet_original = []
    for line in f_in:
        tweet = json.loads(line)
        if tweet.get('retweeted_status'): #this is retweet
            if  len(id_tweet_original)< number_original_tweet_wanted:
                f_info_out.write("{0},{1},{2}\n".format(tweet['user']['id_str'], tweet['retweeted_status']['user']['id_str'],tweet['retweeted_status']['id_str']))
                if (tweet['retweeted_status']['id'] not in id_tweet_original):
                    id_tweet_original.append(tweet['retweeted_status']['id'])
                    f_id_orgininal_tweet_out.write('{0},"{1}"\n'.format(tweet['retweeted_status']['id'], tweet['retweeted_status']['text'].encode('utf-8').replace('\n', '').replace('\r', '')))
                    f_orgininal_tweet_out.write("{0}\n".format(tweet['retweeted_status']['text'].encode('utf-8').replace('\n', '').replace('\r', '')))
            else:
                if (tweet['retweeted_status']['id'] in id_tweet_original): #if original id in [] then record the infor, otherwise ignore
                    f_info_out.write("{0},{1},{2}\n".format(tweet['user']['id_str'], tweet['retweeted_status']['user']['id_str'],tweet['retweeted_status']['id_str']))
    f_in.close()
    f_id_orgininal_tweet_out.close()
    f_orgininal_tweet_out.close()
    f_info_out.close()


def check_number_user(file_in):#each line includes user_retweet,user_original,tweet_id
    f_in = open(file_in,'r')
    id_user_retweet = []
    id_user_original = []
    id_tweet  = []
    for line in f_in:
        user_retweet = line.split(',')[0]
        user_original = line.split(',')[1]
        tweet_id = line.split(',')[2]
        if user_retweet not in id_user_retweet:
            id_user_retweet.append(user_retweet)
        if user_original not in id_user_original:
            id_user_original.append(user_original)
        if tweet_id not in id_tweet:
            id_tweet.append(tweet_id)
    print ('number of user retweet: ',len(id_user_retweet),'\nnumber of user original: ',len(id_user_original),'\nnumber of tweet',len(id_tweet))
    f_in.close()


def get_first_retweetedTweet(file_jason_in, file_UserRetweet_UserOriginal_out, file_UserRetweet_UserOriginal_IdOrginalTweet_out,file_Id_orgininal_tweet_out,file_orgininal_tweet_out,id_orgininal_tweet_json_path,number_Retweet_wanted):
    f_in = open(file_jason_in, 'r')
    f_UserRetweet_UserOriginal_out= open(file_UserRetweet_UserOriginal_out,'w')
    f_UserRetweet_UserOriginal_IdOrginalTweet_out = open(file_UserRetweet_UserOriginal_IdOrginalTweet_out, 'w')
    f_Id_orgininal_tweet_out = open(file_Id_orgininal_tweet_out, 'w')
    f_orgininal_tweet_out = open(file_orgininal_tweet_out, 'w')
    id_tweet_original = []
    id_tweet_text = {}
    number_retweeted_achived = 0

    for line in f_in:
        tweet = json.loads(line)

        if tweet.get('retweeted_status'): #this is retweet
            number_retweeted_achived = number_retweeted_achived + 1
            if number_retweeted_achived <= number_Retweet_wanted:
                f_UserRetweet_UserOriginal_out.write("{0},{1}\n".format(tweet['user']['id_str'], tweet['retweeted_status']['user']['id_str']))
                f_UserRetweet_UserOriginal_IdOrginalTweet_out.write("{0},{1},{2}\n".format(tweet['user']['id_str'], tweet['retweeted_status']['user']['id_str'], tweet['retweeted_status']['id_str']))

                if tweet['retweeted_status']['id'] not in id_tweet_original:
                    id_tweet_original.append(tweet['retweeted_status']['id'])
                    id_tweet_text[tweet['retweeted_status']['id']] = tweet['retweeted_status']['text']
                    f_Id_orgininal_tweet_out.write("{0},{1}\n".format(tweet['retweeted_status']['id'], tweet['retweeted_status']['text']))
                    #f_Id_orgininal_tweet_out.write("{0},{1}\n".format(tweet['retweeted_status']['id'], tweet['retweeted_status']['text'].encode('utf-8').replace('\n', '').replace('\r', '')))
                    f_orgininal_tweet_out.write("{0}\n".format(tweet['retweeted_status']['text']))
                    #f_orgininal_tweet_out.write("{0}\n".format(tweet['retweeted_status']['text'].encode('utf-8').replace('\n', '').replace('\r', '')))
    f_in.close()
    f_UserRetweet_UserOriginal_out.close()
    f_UserRetweet_UserOriginal_IdOrginalTweet_out.close()
    f_Id_orgininal_tweet_out.close()
    f_orgininal_tweet_out.close()
    json_tweet_text = json.dumps(id_tweet_text)

    with open(id_orgininal_tweet_json_path, 'w') as f:
        json.dump(json_tweet_text, f)

#def get_tweets():
#    '''
#        return the tweets and retweets
#    '''
#    get_first_retweetedTweet('1_percent_tweets_second_week_Jan_2017.json','20000_UserRetweet_UserOriginal.txt','20000_UserRetweet_UserOriginal_idOriginalTweet.txt','20000_id_OriginalTweet','20000_Original_tweet.txt',20000)
#

if __name__ == '__main__':

    data_dir = sys.argv[1]
    dataPath = Datapath(data_dir)

    data_raw_dir = dataPath.get_data_raw_dir()
    tweets_raw_path = os.path.join(data_raw_dir, '1_percent_tweets_second_week_Jan_2017.json')

    data_processed_dir = dataPath.get_data_processed_dir()
    userretweet_useroriginal_path = os.path.join(data_processed_dir, '20000_UserRetweet_UserOriginal.txt')
    userretweet_userid_original_path = os.path.join(data_processed_dir, '20000_UserRetweet_UserOriginal_idOriginalTweet.txt')
    id_originaltweet_path = os.path.join(data_processed_dir, '20000_id_OriginalTweet')
    id_orgininal_tweet_json_path = os.path.join(data_processed_dir, '20000_id_OriginalTweet.json')
    original_tweet_path = os.path.join(data_processed_dir, '20000_Original_tweet.txt')

    get_first_retweetedTweet(tweets_raw_path, userretweet_useroriginal_path, userretweet_userid_original_path,
                             id_originaltweet_path, original_tweet_path, id_orgininal_tweet_json_path, 20000)

