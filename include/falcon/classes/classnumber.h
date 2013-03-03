/*
 FALCON - The Falcon Programming Language.
 FILE: classnumber.h
 
 Int object handler.
 -------------------------------------------------------------------
 Author: Francesco Magliocca
 Begin: Sun, 10 Feb 2013 05:08:17 +0100
 
 -------------------------------------------------------------------
 (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)
 
 See LICENSE file for licensing details.
 */

#ifndef _FALCON_CLASSNUMBER_H_
#define _FALCON_CLASSNUMBER_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{
    
/**
Abstract base class for all numbers.
*/

class FALCON_DYN_CLASS ClassNumber: public Class
{
public:

   ClassNumber();
   virtual ~ClassNumber();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
};

}

#endif

/* end of classnumber.h */
