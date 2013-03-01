/*
   FALCON - The Falcon Programming Language.
   FILE: stringstream.h

   Falcon core module -- String stream interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 01 Mar 2013 01:02:02 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_STRINGSTREAM_H
#define FALCON_CORE_STRINGSTREAM_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classstream.h>

#include <falcon/method.h>

namespace Falcon {
namespace Ext {

   /*#
    @class StringStream
    @brief Memory based virtual stream.

    */
class FALCON_DYN_CLASS ClassStringStream: public ClassStream
{
public:
   ClassStringStream();
   virtual ~ClassStringStream();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;

   /*#
    @property pipeMode StringStream
    @brief Gets or sets the pipe mode for this string stream.

    In pipe mode, the string stream read and write pointers are different.

    If the mode is set to false, they move together, and the write pointer
    is reset to the position of the read pointer.

    In pipe mode, seek() moves both the read and the write pointer,
    and current position is relative to write pointer,
    but tell() returns the read pointer.
   */
   FALCON_DECLARE_PROPERTY(pipeMode);

   /*#
    @property content StringStream

    @brief The whole content of this string stream as a memory buffer string.
   */
   FALCON_DECLARE_PROPERTY(content);

   /*#
    @method closeToString StringStream

    @brief Gets the data written to the stream in a memory-efficient way.

    Closes the string and passes the string memory as-is to a memory buffer string.
    After this call the stream is not usable anymore.
   */
   FALCON_DECLARE_METHOD(closeToString, "");

};

}
}

#endif	

/* end of stringstream.h */
