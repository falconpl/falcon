/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.h

   Main inclusion file file
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 25 Jul 2011 15:17:33 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FALCON_H
#define	FALCON_FALCON_H

// This includes the vast majority of things in Falcon.
#include <falcon/engine.h>
#include <falcon/log.h>

#include <falcon/expression.h>
#include <falcon/statement.h>
#include <falcon/psteps/exprrule.h>

#include <falcon/stdstreams.h>
#include <falcon/streambuffer.h>
#include <falcon/stringstream.h>
#include <falcon/textwriter.h>
#include <falcon/textreader.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <falcon/debugger.h>


#include <falcon/module.h>
#include <falcon/intcompiler.h>
#include <falcon/modcompiler.h>
#include <falcon/modspace.h>
#include <falcon/process.h>
#include <falcon/loaderprocess.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/wvmcontext.h>
#include <falcon/application.h>

#include <falcon/cm/coremodule.h>
#include <falcon/vfsprovider.h>

#include <falcon/symbol.h>

#include <falcon/item.h>
#include <falcon/itemid.h>

//--- error headers ---
#include <falcon/classes/classerror.h>

#include <falcon/errors/accesserror.h>
#include <falcon/errors/accesstypeerror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/errors/genericerror.h>
#include <falcon/errors/interruptederror.h>
#include <falcon/errors/ioerror.h>
#include <falcon/errors/operanderror.h>
#include <falcon/errors/unsupportederror.h>
#include <falcon/errors/syntaxerror.h>
#include <falcon/errors/encodingerror.h>
#include <falcon/errors/linkerror.h>


#endif	/* FALCON_H */

/* end of falcon.h */
