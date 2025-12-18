import argparse
import os

from src.data.vocab import Vocab
from src.model.seq2seq import Seq2Seq
from src.model.seq2seq_with_additive_attention import Seq2SeqWithAdditiveAttention
from src.model.seq2seq_with_global_attention import Seq2SeqWithGlobalAttention
from src.training.trainer import Trainer
from src.utils.logging import setup_logger


def run_assignment_1(args):
    logger = setup_logger(
        output=os.path.join(args.log_dir, "assignment_1"),
        name="assignment_1"
    )

    vocab = Vocab(
        data_file=args.train_path,
        src_lang="vietnamese",
        tar_lang="english"
    )

    model = Seq2Seq(
        vocab=vocab,
        embed_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.num_layers,
        dropout=args.dropout
    )

    trainer = Trainer(
        vocab=vocab,
        model=model,
        train_path=args.train_path,
        dev_path=args.val_path,
        test_path=args.test_path,
        logger=logger,
        checkpoint_path=args.checkpoint_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )

    trainer.train(
        num_epochs=args.epochs,
        patience=args.patience
    )

    test_loss, test_rouge = trainer.test()
    logger.info(f"Final Results - Loss: {test_loss:.4f} | ROUGE-L: {test_rouge:.4f}")
    print(f"\n{'='*60}")
    print(f"TEST LOSS: {test_loss:.4f}")
    print(f"TEST ROUGE-L: {test_rouge:.4f}")
    print(f"{'='*60}\n")


def run_assignment_2(args):
    logger = setup_logger(
        output=os.path.join(args.log_dir, "assignment_2"),
        name="assignment_2"
    )

    vocab = Vocab(
        data_file=args.train_path,
        src_lang="vietnamese",
        tar_lang="english"
    )

    model = Seq2SeqWithAdditiveAttention(
        vocab=vocab,
        embed_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.num_layers,
        dropout=args.dropout
    )

    trainer = Trainer(
        vocab=vocab,
        model=model,
        train_path=args.train_path,
        dev_path=args.val_path,
        test_path=args.test_path,
        logger=logger,
        checkpoint_path=args.checkpoint_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )

    trainer.train(
        num_epochs=args.epochs,
        patience=args.patience
    )

    test_loss, test_rouge = trainer.test()
    logger.info(f"Final Results - Loss: {test_loss:.4f} | ROUGE-L: {test_rouge:.4f}")
    print(f"\n{'='*60}")
    print(f"TEST LOSS: {test_loss:.4f}")
    print(f"TEST ROUGE-L: {test_rouge:.4f}")
    print(f"{'='*60}\n")


def run_assignment_3(args):
    logger = setup_logger(
        output=os.path.join(args.log_dir, "assignment_3"),
        name="assignment_3"
    )

    vocab = Vocab(
        data_file=args.train_path,
        src_lang="vietnamese",
        tar_lang="english"
    )

    model = Seq2SeqWithGlobalAttention(
        vocab=vocab,
        embed_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.num_layers,
        dropout=args.dropout,
        attention_type="general"
    )

    trainer = Trainer(
        vocab=vocab,
        model=model,
        train_path=args.train_path,
        dev_path=args.val_path,
        test_path=args.test_path,
        logger=logger,
        checkpoint_path=args.checkpoint_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )

    trainer.train(
        num_epochs=args.epochs,
        patience=args.patience
    )

    test_loss, test_rouge = trainer.test()
    logger.info(f"Final Results - Loss: {test_loss:.4f} | ROUGE-L: {test_rouge:.4f}")
    print(f"\n{'='*60}")
    print(f"TEST LOSS: {test_loss:.4f}")
    print(f"TEST ROUGE-L: {test_rouge:.4f}")
    print(f"{'='*60}\n")


def main(args):
    print(f"\n{'='*60}")
    print(f"Starting Assignment {args.assignment}")
    print(f"{'='*60}\n")
    
    if args.assignment == '1':
        run_assignment_1(args)
    elif args.assignment == '2':
        run_assignment_2(args)
    elif args.assignment == '3':
        run_assignment_3(args)
    else:
        raise ValueError(f"Invalid assignment: {args.assignment}. Choose from ['1', '2', '3']")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--assignment", type=str, default="1", choices=['1', '2', '3'])
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    main(args)
