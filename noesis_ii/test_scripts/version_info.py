"""版本信息管理"""

VERSION = "1.0.0"

def get_version():
    """获取当前版本"""
    return VERSION

def check_version_compatibility(version):
    """检查版本兼容性"""
    current_major = int(VERSION.split('.')[0])
    check_major = int(version.split('.')[0])
    return current_major == check_major