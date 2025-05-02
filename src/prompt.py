from typing import Any

def ask_prompt(prompts: list[tuple[str, callable]], main_prompt: str = "Pick an option:") -> Any:
    print(main_prompt)
    for i, (prompt, _) in enumerate(prompts):
        print(f"{i+1}. {prompt}")

    while(True):
        try:
            choice = int(input("> "))

            if choice < 1 or choice > len(prompts):
                raise IndexError("Invalid choice")
            return prompts[choice-1][1]()
        except KeyboardInterrupt:
            exit(0)
        except ValueError:
            print(f"Please pick a valid choice (1-{len(prompts)})")
        except IndexError:
            print(f"Please pick a valid choice (1-{len(prompts)})")
