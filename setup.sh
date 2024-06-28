mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
[theme]\
base="dark"\
\n\
" > ~/.streamlit/config.toml
