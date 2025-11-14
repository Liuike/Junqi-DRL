# This is the repository for the final project of CSCI1470 at Brown University

# Building Project Environment

This guide explains how to configure an OSCAR environment to develop or run the project

## Step 1: Basic OSCAR Setup

### 1. Enter interact mode and stay in interact for all the following instructions
- failure to do so will result in an OSCAR warning email...

### 2. Install uv (our dependency manager)
```bash
pip install uv
```

## Step 2: Setting up Clang
Before we can actually install the packages, open-spiel(the training environment) requires clang to actually work, and OSCAR uses something that isn't exactly clang, so we need the following setup.

### 1. Verify LLVM Module is Loaded

Check that the LLVM module is loaded:

```bash
module list
```

You should see `llvm/16.0.2-mq6g5lb` in the list.

### 2. Set Compiler Environment Variables

Configure the build system to use LLVM's Clang compiler:

```bash
export CC=$(llvm-config --bindir)/clang
export CXX=$(llvm-config --bindir)/clang++
```

### 3. Verify Compilers are Accessible

Check that the compilers are correctly set:

```bash
$CC --version
$CXX --version
```

Both commands should display Clang version 16.0.2.

## Step 3. Final Env Setup

Now you can install the package using uv:

```bash
uv sync
```

After the command successfully finishes, you should see a .venv folder. Activate it using: 
```bash
source .venv/bin/activate
```

Now, try running the manual play example script to make sure everything is working!
```bash
python examples/junqi_standard_example.py
```

## Important Notes During Development

#### Adding Dependencies

If you want to add any other package, instead of using pip using, do the following: 
```bash
uv add <package>
```
Then, uv will automatically update uv.lock and pyproject.toml. Then pushing both files to github will let everyone else know you added this dependency. 

#### After git pull: 

If you see that pyproject.toml and uv.lock has been changed, make sure to run 
```bash
uv sync
```
again to make sure you are updated with the most recent dependencies. 

## Troubleshooting

### Issue: `clang: command not found`

**Solution**: The LLVM module may not be loaded or the environment variables are not set. Follow steps 1-3 above.

### Issue: `RuntimeError: A C++ compiler that supports c++17 must be installed`

**Solution**: Ensure you've set the `CC` and `CXX` environment variables to point to LLVM's Clang compiler (step 3).