/*
   FALCON - The Falcon Programming Language.
   FILE: falconstate.h

   State for Falcon Classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 19:01:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FALCONSTATE_H_
#define _FALCON_FALCONSTATE_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon
{

/** State for Falcon Classes.
 
 A State is a set of methods that override the standard methods used by an
 instance. When a state is active, methods are taken from the state, and not
 from the base class definition.

 \note The class is fully non-virtual.
 */
class FALCON_DYN_CLASS FalconState
{
public:

   FalconState( const String& name );
   ~FalconState();

   const String& name() const { return m_name; }
   
private:
   const String m_name;

   class Private;
   Private* _p;
};

}

#endif /* _FALCON_FALCONSTATE_H_ */

/* end of falconstate.h */
