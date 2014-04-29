// Copyright 2004 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "re2/stringpiece.h"
#include "util/util.h"

using re2::StringPiece;

std::ostream& operator<<(std::ostream& o, const StringPiece& piece) {
  o.write(piece.data(), piece.size());
  return o;
}

bool StringPiece::_equal(const StringPiece& x, const StringPiece& y) {
  int len = x.size();
  if (len != y.size()) {
    return false;
  }
  const char* p = x.data();
  const char* p2 = y.data();
  // Test last byte in case strings share large common prefix
  if ((len > 0) && (p[len-1] != p2[len-1])) return false;
  const char* p_limit = p + len;
  for (; p < p_limit; p++, p2++) {
    if (*p != *p2)
      return false;
  }
  return true;
}

void StringPiece::CopyToString(string* target) const {
  target->assign(ptr_, length_);
}

int StringPiece::copy(char* buf, size_type n, size_type pos) const {
  int ret = min(length_ - pos, n);
  memcpy(buf, ptr_ + pos, ret);
  return ret;
}

int StringPiece::find(const StringPiece& s, size_type pos) const {
  if (length_ < 0 || pos > static_cast<size_type>(length_))
    return npos;

  const char* result = std::search(ptr_ + pos, ptr_ + length_,
                                   s.ptr_, s.ptr_ + s.length_);
  const size_type xpos = result - ptr_;
  return xpos + s.length_ <= (size_t)length_ ? xpos : npos;
}

int StringPiece::find(char c, size_type pos) const {
  if (length_ <= 0 || pos >= static_cast<size_type>(length_)) {
    return npos;
  }
  const char* result = std::find(ptr_ + pos, ptr_ + length_, c);
  return result != ptr_ + length_ ? result - ptr_ : npos;
}

int StringPiece::rfind(const StringPiece& s, size_type pos) const {
  if (length_ < s.length_) return npos;
  const size_t ulen = length_;
  if (s.length_ == 0) return min(ulen, pos);

  const char* last = ptr_ + min(ulen - s.length_, pos) + s.length_;
  const char* result = std::find_end(ptr_, last, s.ptr_, s.ptr_ + s.length_);
  return result != last ? result - ptr_ : npos;
}

int StringPiece::rfind(char c, size_type pos) const {
  if (length_ <= 0) return npos;
  for (int i = min(pos, static_cast<size_type>(length_ - 1));
       i >= 0; --i) {
    if (ptr_[i] == c) {
      return i;
    }
  }
  return npos;
}

StringPiece StringPiece::substr(size_type pos, size_type n) const {
  if (pos > (size_type) length_) pos = length_;
  if (n > length_ - pos) n = length_ - pos;
  return StringPiece(ptr_ + pos, n);
}

const StringPiece::size_type StringPiece::npos = size_type(-1);

void StringPiece::set( const Falcon::String& src, Falcon::length_t start )
{
   // check if we can use the string data directly.

   /*
   This pre-check is all less efficient than transforming the string in utf8 unconditionally,
   however, if we ever have this check preloaded in the string, we might use it.
   For this reason I leave the stuff commented here.

   if( src.manipulator()->charSize() == 1 )
   {
      Falcon::byte* pos = src.getRawStorage() + start;
      Falcon::byte* end = pos+src.size();

      while( pos < end )
      {
         if ( *pos >= 128 )
         {
            break;
         }
         ++pos;
      }

      if( pos == end )
      {
         // cool we can proceed with the string as-is
         buffer_ = 0;
         ptr_ = reinterpret_cast<char*>(src.getRawStorage());
         length_ = src.size();
         return;
      }
   }
   */

   // we must convert to utf-8
   Falcon::length_t bufSize = src.length()-start;
   Falcon::length_t result = bufSize;
   do
   {
      delete[] buffer_; // initally 0
      bufSize = bufSize * 2 + 4;
      buffer_ = new char[bufSize];
      if( start == 0 )
      {
         result = src.toUTF8String( buffer_, bufSize );
      }
      else {
         result = src.subString(start).toUTF8String(buffer_, bufSize );
      }
   }
   while( result == Falcon::String::npos );

   ptr_ = buffer_;
   length_ = result;
}
