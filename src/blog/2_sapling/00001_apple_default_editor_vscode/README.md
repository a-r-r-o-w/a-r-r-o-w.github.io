---
{
  "title": "Setting VS Code as the default editor on MacOS",
  "authors": ["Aryan V S"],
  "date": "2025-11-11",
  "tags": ["resources", "apple"]
}
---

If you like to use VS Code like me, your probably want to open all your text/code files with it. On my machine, by default, most file extensions open with XCode. This was the case with `Command + Click`, keyboard shortcuts, and `open <FILENAME>` in the terminal. XCode is great but I don't fancy using it, so this behavior was quite annoying.

One solution to open a specific file extension with VS Code is to do the standard `Right Click -> Open With -> Select Always Open With -> Select app`. This is still, however, quite annoying to do for every file extension type.

There's fortunately a better approach (thanks Claude!):
- Use AppleScript to find the identifier of your VS Code app: `osascript -e 'id of app "Visual Studio Code"'`. In my case, this is `com.microsoft.VSCode`
- Install [duti](https://github.com/moretension/duti): `brew install duti`. This helps in easily mapping different file types to the editor/default app of your choice.
- Modify the following script to map the file extensions of your choice:
```bash
#!/bin/bash

public_types=(
  public.plain-text
  public.python-script
  public.shell-script
  public.source-code
  public.text
  # For files without an extension
  public.data
)

for type in "${public_types[@]}"; do
  duti -s com.microsoft.VSCode "$type" all
done

extensions=(
  .c .cpp .h .json .log .md .toml .txt .conf .yaml .yml
)

for ext in "${extensions[@]}"; do
  duti -s com.microsoft.VSCode "$ext" all
done
```

If your git merge/rebase commit confirmations, or other files that you expect to open in vim, start opening in VS Code, configure git to use vim:
```bash
git config --global core.editor vim
```

Hope I saved you some headache!
