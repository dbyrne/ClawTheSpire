from enum import Enum


class CardType(str, Enum):
    ATTACK = "Attack"
    SKILL = "Skill"
    POWER = "Power"
    STATUS = "Status"
    CURSE = "Curse"


class TargetType(str, Enum):
    SELF = "Self"
    ANY_ENEMY = "AnyEnemy"
    ALL_ENEMIES = "AllEnemies"
    RANDOM_ENEMY = "RandomEnemy"
    ANY_ALLY = "AnyAlly"


class CardZone(str, Enum):
    HAND = "hand"
    DRAW_PILE = "draw_pile"
    DISCARD_PILE = "discard_pile"
    EXHAUST_PILE = "exhaust_pile"
