# coding=utf-8

from migration.loader import MigrationLoader

if __name__ == '__main__':
    loader = MigrationLoader()
    changes = loader.make_plan()
    for change in changes:
        print(change.key, change.parent)
