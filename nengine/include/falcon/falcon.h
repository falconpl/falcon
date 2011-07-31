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

#include <falcon/expression.h>
#include <falcon/statement.h>
#include <falcon/stmtautoexpr.h>
#include <falcon/stmtrule.h>
#include <falcon/stmt_init.h>

#include <falcon/stdstreams.h>
#include <falcon/textwriter.h>
#include <falcon/textreader.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <falcon/module.h>
#include <falcon/intcompiler.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/application.h>

#include <falcon/cm/coremodule.h>

#include <falcon/vfsprovider.h>


//--- error headers ---
#include <falcon/accesserror.h>
#include <falcon/accesstypeerror.h>
#include <falcon/classerror.h>
#include <falcon/codeerror.h>
#include <falcon/genericerror.h>
#include <falcon/interruptederror.h>
#include <falcon/ioerror.h>
#include <falcon/operanderror.h>
#include <falcon/unsupportederror.h>
#include <falcon/syntaxerror.h>
#include <falcon/encodingerror.h>

#endif	/* FALCON_H */

/* end of falcon.h */
