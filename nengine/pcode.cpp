/*
   FALCON - The Falcon Programming Language.
   FILE: pcode.cpp

   Falcon virtual machine - pre-compiled code
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Jan 2011 17:54:13 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/pcode.h>
#include <falcon/vm.h>

namespace Falcon {

void PCode::apply( VMachine* vm ) const
{
	CodeFrame& cf = vm->currentCode();
	// TODO Check if all this loops are really performance wise
	register int depth = vm->codeDepth();
	register int pos = m_steps.size() - cf.m_seqId;
	while ( pos > 0 && vm->codeDepth() == depth )
	{
		const PStep* pstep = m_steps[ --pos ];
		pstep->apply(vm);
	}

	// we're done?
	if( pos == 0 )
	{
	   vm->popCode();
	   // save the result in the A register
	   vm->regA() = vm->topData();
	   vm->popData();
	}
	else
	{
	   cf.m_seqId = m_steps.size() - pos;
	}
}

}

/* end of pcode.cpp */
