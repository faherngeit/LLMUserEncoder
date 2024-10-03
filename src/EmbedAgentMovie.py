from functools import cache
from textwrap import dedent

from EmbedAgent import EmbedAgent
from movie.movie_user import MovieUser


class EmbedAgentMovie(EmbedAgent):
    """A class used to interact with the embedding agent specifically for movie users.

    Methods:
        get_user_description(user: MovieUser, test: bool = False): Gets the user description.
    """

    @cache
    def get_user_description(self, user: MovieUser, test: bool = False) -> str | list:
        """Gets the user description.

        Args:
            user (MovieUser): The user to get the description for.
            test (bool, optional): Whether to run in test mode. Defaults to False.

        Returns:
            str | list: The description of the user or the prompt if in test mode.
        """
        prompt = [
            {
                "role": "system",
                "content": dedent(
                    """
                    You will be presented with user ratings and your job is to provide a general
                    summarization of users preferences. You should pay attention to movie genres, its release year,
                    main cast, director, awards, critical acclaims, and other relevant information.
                    The response should be split into several parts: first should provide the analysis of movie
                    preferences based on its year. The second should be focused on genres and plot twists.
                    The third one should describe preferences in cast and directors.
                    The fourth should characterise users choice based on correlation between users ranking and critical
                    acclamation and reviews of movies. The fifth paragraph should describe the movies with user dislike
                    based on assigned rating 3 and below. What do these low-ranked movies have in common?
                    Provided user's profile as a general description of the user's preferences should avoid mentioning
                    the actual movies and ratings or user personal data. Try to predict favourite users actor, director,
                    genre, etc. The user profile should be useful for further movie recommendations and it should be at
                    least 5 paragraphs long. The description should include characteristics of the most relevant genres
                    to the user. Avoid mentioning specific movies in response.
                    """).replace("\n", " ")
            },
            {
                "role": "user",
                "content": user.prompt()
            }
        ]
        if test:
            return prompt
        return self.agent.send_prompt(prompt).choices[0].message.content
