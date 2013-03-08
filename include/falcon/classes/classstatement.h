/*
   FALCON - The Falcon Programming Language.
   FILE: classstatement.h

   Base class for statement PStep handlers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Dec 2011 21:39:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSSTATEMENT_H_
#define _FALCON_CLASSSTATEMENT_H_

#include <falcon/setup.h>
#include <falcon/classes/classtreestep.h>

namespace Falcon {

class ClassTreeStep;

/** Handler class for Statement class.
 
 This handler manages the base statements as they are reflected into scripts,
 and has also support to handle the vast majority of serialization processes.
 
 */
class ClassStatement: public ClassTreeStep
{
public:
   ClassStatement( ClassTreeStep* parent );
   virtual ~ClassStatement();

protected:
   ClassStatement( const String& name );
};

}

#endif 

/* end of classstatement.h */
