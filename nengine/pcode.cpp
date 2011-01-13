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

void PCode::perform( VMachine* vm ) const
{
	vm->pushCode(this);
	CodeFrame& cf = vm->currentCode();
	cf.m_seqId = size();
}

void PCode::apply( VMachine* vm ) const
{
	CodeFrame& cf = vm->currentCode();
	// TODO Check if all this loops are really performance wise
	int depth = vm->codeDepth();
	while ( cf.m_seqId > 0 && vm->codeDepth() == depth )
	{
		const PStep* pstep = m_steps[ --cf.m_seqId ];
		pstep->apply(vm);
	}

	// we're done?
	if(  cf.m_seqId == 0 )
	   vm->popCode();
}

}

/* end of pcode.cpp */
