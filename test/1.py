"""
用python设计第一个游戏
"""

def main():
    print("欢迎来到我的游戏！")
    print("游戏规则：")
    print("你将猜测我现在心里想的是哪个数字，我会告诉你你猜对了没有。")
    print("游戏开始！")
    guess_number()

def guess_number():
    temp:str = input("不妨猜想我现在的心里在想的是什么数字,")
    guess:str = temp.strip()
    if guess == '8':
        print("你是我心理的蛔虫嘛？！")
        print("哼，猜中了也没有奖励！")
    else:
        print("猜错啦，我现在心理想的是8！")
    print("游戏结束，不玩啦*￣︶￣")

if __name__ == '__main__':
    main()
