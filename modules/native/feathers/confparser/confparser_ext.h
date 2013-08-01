/*
   FALCON - The Falcon Programming Language.
   FILE: socket_ext.cpp

   Falcon VM interface to confparser module -- header.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon VM interface to confparser module -- header.
*/


#ifndef FLC_CONFPARSER_EXT_H
#define FLC_CONFPARSER_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>

#include <falcon/error_base.h>

#ifndef FALCON_CONFPARSER_ERROR_BASE
   #define FALCON_CONFPARSER_ERROR_BASE        1110
#endif

#define FALCP_ERR_INVFORMAT  (FALCON_CONFPARSER_ERROR_BASE + 0)
#define FALCP_ERR_STORE      (FALCON_CONFPARSER_ERROR_BASE + 1)

namespace Falcon {
namespace Ext {

// ==============================================
// Class ConfParser
// ==============================================

class ClassConfParser: public ::Falcon::Class
{
public:
   ClassConfParser();
   virtual ~ClassConfParser();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;
};


Class* confparser_create();
}
}

#endif

/* end of socket_ext.h */
