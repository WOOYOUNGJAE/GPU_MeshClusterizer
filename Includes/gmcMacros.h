#pragma once

#define STORE_UINT16_IN_UINT32(TARGET, IS_HIGHER, VAL) \
    (TARGET) = ((IS_HIGHER) != 0) \
        ? ((TARGET) & 0x0000FFFF) | (((unsigned int)(VAL) & 0xFFFF) << 16) \
        : ((TARGET) & 0xFFFF0000) | ((unsigned int)(VAL) & 0xFFFF)

#define LOAD_UINT16_FROM_UINT32(TARGET, IS_HIGHER) \
    (unsigned short)(((IS_HIGHER) != 0) \
        ? ((TARGET) >> 16) & 0xFFFF \
        : (TARGET) & 0xFFFF)

#define MARK_MSB_UINT32(TARGET, MARK) \
    ((TARGET) |= ((unsigned int)((MARK) != 0) << 31))

#define UNPACK_MSB_UINT32(TARGET) \
    ((TARGET >> 31) & 1u)

#define IGNORE_MSB_UINT32(TARGET) \
    ((TARGET) & ~0x8000'0000u)
