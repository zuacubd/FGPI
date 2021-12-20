#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import json
import os
import sys
from app.path_manager import Datapath


def extract_text_from_tweet_json(file_in_json, file_out_text):
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

    line = f_in.readline()
    tweets = json.loads(line)
    tweets_post = tweets['hits']['hits']

    for tweet in tweets_post:

        if tweet['fields']["relation_type"] == "Mention": #this is a reply
            if  len(id_tweet_original)< number_original_tweet_wanted:
                f_info_out.write("{0},{1},{2}\n".format(tweet['fields']['creator_Id'], tweet['fields']['ref_creator_id'],tweet['fields']['id']))
                if (tweet['fields']['id'] not in id_tweet_original):
                    id_tweet_original.append(tweet['fields']['id'])
                    f_id_orgininal_tweet_out.write('{0},"{1}"\n'.format(tweet['fields']['id'], tweet['fields']['attr.content'].encode('utf-8').replace('\n', '').replace('\r', '')))
                    f_orgininal_tweet_out.write("{0}\n".format(tweet['fields']['attr.content'].encode('utf-8').replace('\n', '').replace('\r', '')))
            else:
                if (tweet['fields']['id'] in id_tweet_original): #if id in [] then record the info, otherwise ignore
                    f_info_out.write("{0},{1},{2}\n".format(tweet['fields']['creator_Id'], tweet['fields']['ref_creator_id'], tweet['fields']['id']))
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


def get_first_retweeted_tweet(file_json_in, file_UserRetweet_UserOriginal_out, file_UserRetweet_UserOriginal_IdOrginalTweet_out,file_Id_orgininal_tweet_out,file_orgininal_tweet_out,id_orgininal_tweet_json_path,number_Retweet_wanted):
    '''
        extracting tweets
    '''
    f_in = open(file_json_in, 'r')
    f_UserRetweet_UserOriginal_out= open(file_UserRetweet_UserOriginal_out, 'w')
    f_UserRetweet_UserOriginal_IdOrginalTweet_out = open(file_UserRetweet_UserOriginal_IdOrginalTweet_out, 'w')
    f_Id_orgininal_tweet_out = open(file_Id_orgininal_tweet_out, 'w')
    f_orgininal_tweet_out = open(file_orgininal_tweet_out, 'w')

    id_tweet_original = []
    id_tweet_text = {}
    number_retweeted_achived = 0

    line = f_in.readline()
    tweets = json.loads(line)
    tweets_post = tweets['hits']['hits']

    for tweet in tweets_post:
        if tweet['fields']["relation_type"] == "Reply": #this is retweet
            number_retweeted_achived = number_retweeted_achived + 1
            if number_retweeted_achived <= number_Retweet_wanted:
                f_UserRetweet_UserOriginal_out.write("{0},{1}\n".format(tweet['fields']['ref_creator_id'], tweet['fields']['creator_Id']))
                f_UserRetweet_UserOriginal_IdOrginalTweet_out.write("{0},{1},{2}\n".format(tweet['fields']['ref_creator_id'], tweet['fields']['creator_Id'], tweet['fields']['id']))

                if tweet['fields']['id'] not in id_tweet_original:
                    id_tweet_original.append(tweet['fields']['id'])
                    id_tweet_text[tweet['fields']['id']] = tweet['fields']['attr.content']
                    f_Id_orgininal_tweet_out.write("{0},{1}\n".format(tweet['fields']['creator_Id'], tweet['fields']['attr.content']))
                    f_orgininal_tweet_out.write("{0}\n".format(tweet['fields']['attr.content']))
    f_in.close()
    f_UserRetweet_UserOriginal_out.close()
    f_UserRetweet_UserOriginal_IdOrginalTweet_out.close()
    f_Id_orgininal_tweet_out.close()
    f_orgininal_tweet_out.close()
    json_tweet_text = json.dumps(id_tweet_text)

    with open(id_orgininal_tweet_json_path, 'w') as f:
        json.dump(json_tweet_text, f)


if __name__ == '__main__':

    data_dir = sys.argv[1]
    dataPath = Datapath(data_dir)

    data_raw_dir = dataPath.get_data_raw_dir()
    tweets_raw_path = os.path.join(data_raw_dir, 'dump0.json')

    data_processed_dir = dataPath.get_data_processed_dir()
    userretweet_useroriginal_path = os.path.join(data_processed_dir, 'UserRetweet_UserOriginal.txt')
    userretweet_userid_original_path = os.path.join(data_processed_dir, 'UserRetweet_UserOriginal_idOriginalTweet.txt')
    id_originaltweet_path = os.path.join(data_processed_dir, 'id_OriginalTweet')
    id_orgininal_tweet_json_path = os.path.join(data_processed_dir, 'id_OriginalTweet.json')
    original_tweet_path = os.path.join(data_processed_dir, 'Original_tweet.txt')

    get_first_retweeted_tweet(tweets_raw_path, userretweet_useroriginal_path, userretweet_userid_original_path,
                             id_originaltweet_path, original_tweet_path, id_orgininal_tweet_json_path, 20000)

