/*
   FALCON - The Falcon Programming Language.
   FILE: classstringstream.h

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
#include <falcon/class.h>

namespace Falcon {


class FALCON_DYN_CLASS ClassStringStream: public ClassStream
{
public:
   ClassStringStream();
   virtual ~ClassStringStream();

   //=============================================================
   //
   virtual void* createInstance() const;

};

}

#endif	

/* end of classstringstream.h */
