/*
   FALCON - The Falcon Programming Language.
   FILE: tokenizer.cpp

   String tokenizer general purpose class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Apr 2014 15:01:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MULTITOKENIZER_H_
#define _FALCON_MULTITOKENIZER_H_

#include <falcon/class.h>

namespace Falcon {
namespace Ext {

class ClassMultiTokenizer: public Class
{
public:
   /** Create the textstream class.
    \param clsStream A ClassStream that will be known in the owning module.
    */
   ClassMultiTokenizer();
   virtual ~ClassMultiTokenizer();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   //=============================================================
   //
   virtual void* createInstance() const;
   void op_iter( VMContext* ctx, void* self ) const;
   void op_next( VMContext* ctx, void* instance ) const;

private:
   Function* m_mthNext;

};
}

}


#endif /* _FALCON_TOKENIZER_H_ */

/* tokenizer.cpp */
