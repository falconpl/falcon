#ifndef ENDIANSWAP_H
#define ENDIANSWAP_H

namespace Falcon {

// swapping function specialized for integer types
template <typename T> void swapI ( T& a, T& b )
{
    a ^= b;
    b ^= a;
    a ^= b;
}

template <size_t T> inline void endianswap_recursive(char *val)
{
    swapI<char>(*val, *(val + T - 1));
    endianswap_recursive<T - 2>(val + 1);
}

template <> inline void endianswap_recursive<0>(char *) {}
template <> inline void endianswap_recursive<1>(char *) {}

template <typename T> inline void endianswap(T& val)
{
    endianswap_recursive<sizeof(T)>((char*)(&val));
}

#if FALCON_LITTLE_ENDIAN == 1
template<typename T> inline void ToBigEndian(T& val) { endianswap<T>(val); }
template<typename T> inline void ToLittleEndian(T&) { }
#else
template<typename T> inline void ToBigEndian(T&) { }
template<typename T> inline void ToLittleEndian(T& val) { endianswap<T>(val); }
#endif

// always convert from little to big endian and vice versa
template<typename T> inline void ToOtherEndian(T& val) { endianswap<T>(val); }

// prevent endian-converting pointers by accident
template<typename T> void EndianConvert(T*);
template<typename T> void EndianConvertReverse(T*);

// no-ops, for speed
inline void ToBigEndian(uint8&) { }
inline void ToBigEndian(int8&)  { }
inline void ToLittleEndian(uint8&) { }
inline void ToLittleEndian(int8&) { }
inline void ToOtherEndian(uint8&) { }
inline void ToOtherEndian(int8&) { }

} // end namespace Falcon

#endif
