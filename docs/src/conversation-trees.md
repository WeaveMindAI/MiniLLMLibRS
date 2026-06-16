# Conversation Trees

A conversation is a tree of `ChatNode` handles. Each node holds one `Message`;
children are alternate continuations. A linear chat is just a tree with one
branch. Holding any node keeps its whole ancestor chain (and the tree) alive.

## Building a thread

`add_user` / `add_assistant` each return the new node, so you chain them:

```rust,no_run
# use minillmlib::ChatNode;
let root = ChatNode::root("You are a terse assistant.");
let leaf = root
    .add_user("What's the capital of France?")
    .add_assistant("Paris.")
    .add_user("And Germany?")
    .add_assistant("Berlin.")
    .add_user("And Italy?"); // the turn we want answered
```

`leaf.thread()` is the full `[system, user, assistant, user, assistant, user]`
message list from root to `leaf`.

## Completing from any node

`node.complete(generator, params)` uses `node`'s root-to-node path as the prompt
and appends the reply as a child of `node`, returning the new assistant node.

```rust,no_run
# use minillmlib::{ChatNode, GeneratorInfo};
# async fn run(leaf: ChatNode, gen: GeneratorInfo) -> minillmlib::Result<()> {
let answer = leaf.complete(&gen, None).await?; // None = default per-request params
println!("{}", answer.message.text().unwrap_or(""));
# Ok(()) }
```

You can complete from any node, not just the leaf, to branch off it. The whole
root-to-that-node path is the context.

## Prebuilt history from a `Vec<Message>`

When you already have a message list, `from_messages` builds the linear chain and
hands back `(root, leaf)`. Complete from the leaf.

```rust,no_run
use minillmlib::{ChatNode, GeneratorInfo, Message};

# async fn run(gen: GeneratorInfo) -> minillmlib::Result<()> {
let history = vec![
    Message::system("You are a terse assistant."),
    Message::user("What's the capital of France?"),
    Message::assistant("Paris."),
    Message::user("And Germany?"),
    Message::assistant("Berlin."),
    Message::user("And Italy?"),
];

let (_root, leaf) = ChatNode::from_messages(&history)?;
let answer = leaf.complete(&gen, None).await?;
# Ok(()) }
```

## Ownership: keep a handle

The tree lives in a shared arena that stays alive as long as you hold **any**
handle into it. `from_messages` returns both `root` and `leaf` precisely so you
don't accidentally drop the only handle. When you chain `add_user`/`add_assistant`,
holding the final node is enough (it keeps its whole ancestor chain). Drop every
handle and the thread is freed.

## Saving and loading threads

```rust,no_run
# use minillmlib::ChatNode;
# fn run(leaf: ChatNode) -> minillmlib::Result<()> {
leaf.save_thread("conversation.json")?;
let (root, leaf) = ChatNode::from_thread_file("conversation.json")?;
# Ok(()) }
```
