"""
Author: Abraham R.

Description: 
This module serves as a simple use case of Redis as conversational history.

"""

import redis
import redis.exceptions 
import json

REDIS_CONVERSATION_KEY = "chatbot:conversation_history"

class RedisClient():
    """
    Redis client class abstracts caching implementation for conversational use with an LLM.
    """
    def __init__(self, host : str, port : int, key = REDIS_CONVERSATION_KEY) -> None:
        self.host = host
        self.port = port 
        self.key = key
        self.client = self.setup()
        if self.client is None :
            raise Exception("failed to connect to Redis")


    def setup(self) -> redis.StrictRedis:
        """
        Sets a redis client and checks for connection
        """
        redis_client = redis.StrictRedis(host=self.host, port=self.port, db=0, decode_responses=True)
        try: 
            if redis_client.ping(): return redis_client

        except (redis.exceptions.ConnectionError):
            print("couldn't connect to Redis")
            return None

    def add_to_conversation(self,role, content):
        """
        cache message according to the role.
        """
        
        message = {"role": role, "content": content}
        self.client.rpush(REDIS_CONVERSATION_KEY, json.dumps(message))


    def get_conversation_history(self):
        """
        Retrieve the list of messages from Redis

        """
        messages = self.client.lrange(REDIS_CONVERSATION_KEY, 0, -1)

        return [json.loads(msg) for msg in messages]
    
    def delete_conversation(self) -> None:
        """
        Deletes the entire conversation history
        """
        self.client.delete(self.key)

    def delete_message(self, role: str, content: str) -> None:
        """
        Deletes an specific message from the conversation history.
        """

        messages = self.client.lrange(self.key, 0, -1)
        for msg in messages:
            message = json.loads(msg)

            if message.get("role") == role and message.get("content") == content:
                self.client.lrem(self.key, 0, msg)  
                break

# Test
if __name__ == "__main__":
    try:
        client = RedisClient("localhost", 6379)
        

        client.add_to_conversation("client", "hello")
        client.add_to_conversation("bot", "how can I assist you?")
        

        print("Conversation History:", client.get_conversation_history())
        

        client.delete_message("client", "hello")
        print("Conversation History after deleting 'hello':", client.get_conversation_history())
        

        client.delete_conversation()
        print("Conversation History after deleting all:", client.get_conversation_history())
        
    except Exception as e:
        print(f"Error: {e}")