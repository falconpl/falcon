/*
   FALCON - The Falcon Programming Language.
   FILE: mxml_mod.h

   Minimal XML processor module - module specific extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Mar 2008 14:44:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Minimal XML processor module - module specific extensions.
*/

#ifndef FALCON_MXML_MOD
#define FALCON_MXML_MOD

#include <falcon/setup.h>
#include <falcon/falcondata.h>
#include "mxml_node.h"

namespace Falcon{
class CoreObject;

namespace Ext{

class NodeCarrier: public Falcon::FalconData
{
   MXML::Node *m_node;

public:

   NodeCarrier( MXML::Node *node, CoreObject *co ):
      m_node( node )
   {
      node->shell( co );
   }
   ~NodeCarrier();

   MXML::Node *node() const { return m_node; }
   virtual FalconData *clone() const;
   virtual void gcMark( uint32 mk ){};

   // just a proxy
   CoreObject *shell() const { return m_node->shell(); }
};

}
}
#endif

/* end of mxml_mod.h */
