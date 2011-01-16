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

PCode::PCode()
{
   apply = apply_;
}

void PCode::apply_( const PStep* self, VMachine* vm )
{
   const StepList& steps = static_cast<const PCode*>(self)->m_steps;
	CodeFrame& cf = vm->currentCode();

	// TODO Check if all this loops are really performance wise
	register int depth = vm->codeDepth();
	register int pos = steps.size() - cf.m_seqId;
	while ( pos > 0 )
	{
		const PStep* pstep = steps[ --pos ];
		pstep->apply(pstep,vm);

		if( vm->codeDepth() != depth )
		{
		   cf.m_seqId = steps.size() - pos;
		   return;
		}
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
	   cf.m_seqId = steps.size() - pos;
	}
}

}

/* end of pcode.cpp */
