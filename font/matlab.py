import matplotlib.font_manager as fm

# 列出所有可用的字体，并过滤掉可能导致问题的字体
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
valid_fonts = []
for font in font_list:
    try:
        prop = fm.FontProperties(fname=font)
        print(prop.get_name(), font)
        valid_fonts.append(font)
    except RuntimeError:
        print(f"Failed to load font: {font}")

# 打印有效的字体列表
print("Valid fonts:")
for font in valid_fonts:
    print(font)
