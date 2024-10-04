import argparse
import json
from tqdm import tqdm
import logging

from src.movie.movie_dataset import MovieDataset
from src.EmbedAgentMovie import EmbedAgentMovie

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate embeddings")
    parser.add_argument("--folder", '-f',
                        type=str,
                        dest='folder',
                        default="data/ml-1m",
                        help="Folder with dataset")
    parser.add_argument("--agent", '-a',
                        type=str,
                        dest='agent',
                        default="openai",
                        help="Agent for LLM model, only openai is supported")
    parser.add_argument("--test", '-t',
                        dest='test',
                        action='store_true',
                        help="Test mode, only generate descriptions without LLM calls")
    parser.add_argument("--result-folder", '-r',
                        dest='result_folder',
                        default="embeddings",
                        help="Folder for results json")
    return parser.parse_args()


def evaluate_embeddings(folder: str,
                        agent: str,
                        test: bool,
                        result_folder: str):
    """
    Evaluates embeddings for users in the dataset.

    Args:
        folder (str): The folder containing the dataset.
        agent (str): The agent for the LLM model.
        test (bool): Whether to run in test mode.
        result_folder (str): The folder for the results.
    """
    dataset = MovieDataset(folder=folder)
    llm_agent = EmbedAgentMovie(agent=agent)
    description_list = []
    error_list = {}
    for user in tqdm(dataset):
        try:
            user.description = llm_agent.get_user_description(user=user, test=test)
            if not test:
                user.embedding = llm_agent.encode_description(user.description)
            description_list.append(user.model_dump(exclude=["AGE_DICT"]))
        except Exception as e:
            error_list[user.id] = str(e)
            logging.error(f"Error processing user {user.id}:\n{e}")

        break

    with open(f"{result_folder}/description.json", 'w') as f:
        json.dump(description_list, f, indent=2)
        logging.info(f"Descriptions saved to {result_folder}/description.json")

    if error_list:
        with open(f"{result_folder}/errors.json", 'w') as f:
            json.dump(error_list, f, indent=2)
            logging.warning(f"Number of Errors occurred is {len(error_list)}, please, see {result_folder}/errors.json")



if __name__ == '__main__':
    """
    Main entry point for the script.
    """
    args = parse_args()
    evaluate_embeddings(args.folder, args.agent, args.test, args.result_folder)
