# Instructions

1) **Installing required dependencies:**
```bash
    pip install -r requirements.txt
```

2) **Сlone the repository, which can be explored by following the link in the subdirectory (deep_sort_realtime):**
```bash
    git clone https://github.com/levan92/deep_sort_realtime.git
```

3) **To import the module correctly, go to the cloned repository and run the command:**
```bash
    pip install .
```

4) **The application should be launched through the terminal indicating the necessary parameters:**
```bash
    - `-p`, `--path`: path to file or directory
    - `-n`, `--num`: camera number (int)
```
## Examples of using

1. **Specifying the path:**
    ```bash
    python script.py -p /folder/file.mp4
    ```
2. **Specifying the number:**
    ```bash
    python script.py -n 0
    ```