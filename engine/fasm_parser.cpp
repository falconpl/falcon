/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse fasm_parse
#define yylex   fasm_lex
#define yyerror fasm_error
#define yylval  fasm_lval
#define yychar  fasm_char
#define yydebug fasm_debug
#define yynerrs fasm_nerrs


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     NAME = 259,
     COMMA = 260,
     COLON = 261,
     DENTRY = 262,
     DGLOBAL = 263,
     DVAR = 264,
     DCONST = 265,
     DATTRIB = 266,
     DLOCAL = 267,
     DPARAM = 268,
     DFUNCDEF = 269,
     DENDFUNC = 270,
     DFUNC = 271,
     DMETHOD = 272,
     DPROP = 273,
     DPROPREF = 274,
     DCLASS = 275,
     DCLASSDEF = 276,
     DCTOR = 277,
     DENDCLASS = 278,
     DFROM = 279,
     DEXTERN = 280,
     DMODULE = 281,
     DLOAD = 282,
     DIMPORT = 283,
     DALIAS = 284,
     DSWITCH = 285,
     DSELECT = 286,
     DCASE = 287,
     DENDSWITCH = 288,
     DLINE = 289,
     DSTRING = 290,
     DISTRING = 291,
     DCSTRING = 292,
     DHAS = 293,
     DHASNT = 294,
     DINHERIT = 295,
     DINSTANCE = 296,
     SYMBOL = 297,
     EXPORT = 298,
     LABEL = 299,
     INTEGER = 300,
     REG_A = 301,
     REG_B = 302,
     REG_S1 = 303,
     REG_S2 = 304,
     REG_L1 = 305,
     REG_L2 = 306,
     NUMERIC = 307,
     STRING = 308,
     STRING_ID = 309,
     TRUE_TOKEN = 310,
     FALSE_TOKEN = 311,
     I_LD = 312,
     I_LNIL = 313,
     NIL = 314,
     I_ADD = 315,
     I_SUB = 316,
     I_MUL = 317,
     I_DIV = 318,
     I_MOD = 319,
     I_POW = 320,
     I_ADDS = 321,
     I_SUBS = 322,
     I_MULS = 323,
     I_DIVS = 324,
     I_POWS = 325,
     I_INC = 326,
     I_DEC = 327,
     I_INCP = 328,
     I_DECP = 329,
     I_NEG = 330,
     I_NOT = 331,
     I_RET = 332,
     I_RETV = 333,
     I_RETA = 334,
     I_FORK = 335,
     I_PUSH = 336,
     I_PSHR = 337,
     I_PSHN = 338,
     I_POP = 339,
     I_LDV = 340,
     I_LDVT = 341,
     I_STV = 342,
     I_STVR = 343,
     I_STVS = 344,
     I_LDP = 345,
     I_LDPT = 346,
     I_STP = 347,
     I_STPR = 348,
     I_STPS = 349,
     I_TRAV = 350,
     I_TRAN = 351,
     I_TRAL = 352,
     I_IPOP = 353,
     I_XPOP = 354,
     I_GENA = 355,
     I_GEND = 356,
     I_GENR = 357,
     I_GEOR = 358,
     I_JMP = 359,
     I_IFT = 360,
     I_IFF = 361,
     I_BOOL = 362,
     I_EQ = 363,
     I_NEQ = 364,
     I_GT = 365,
     I_GE = 366,
     I_LT = 367,
     I_LE = 368,
     I_UNPK = 369,
     I_UNPS = 370,
     I_CALL = 371,
     I_INST = 372,
     I_SWCH = 373,
     I_SELE = 374,
     I_NOP = 375,
     I_TRY = 376,
     I_JTRY = 377,
     I_PTRY = 378,
     I_RIS = 379,
     I_LDRF = 380,
     I_ONCE = 381,
     I_BAND = 382,
     I_BOR = 383,
     I_BXOR = 384,
     I_BNOT = 385,
     I_MODS = 386,
     I_AND = 387,
     I_OR = 388,
     I_ANDS = 389,
     I_ORS = 390,
     I_XORS = 391,
     I_NOTS = 392,
     I_HAS = 393,
     I_HASN = 394,
     I_GIVE = 395,
     I_GIVN = 396,
     I_IN = 397,
     I_NOIN = 398,
     I_PROV = 399,
     I_END = 400,
     I_PEEK = 401,
     I_PSIN = 402,
     I_PASS = 403,
     I_SHR = 404,
     I_SHL = 405,
     I_SHRS = 406,
     I_SHLS = 407,
     I_LDVR = 408,
     I_LDPR = 409,
     I_LSB = 410,
     I_INDI = 411,
     I_STEX = 412,
     I_TRAC = 413,
     I_WRT = 414,
     I_STO = 415,
     I_FORB = 416,
     I_EVAL = 417,
     I_OOB = 418
   };
#endif
/* Tokens.  */
#define EOL 258
#define NAME 259
#define COMMA 260
#define COLON 261
#define DENTRY 262
#define DGLOBAL 263
#define DVAR 264
#define DCONST 265
#define DATTRIB 266
#define DLOCAL 267
#define DPARAM 268
#define DFUNCDEF 269
#define DENDFUNC 270
#define DFUNC 271
#define DMETHOD 272
#define DPROP 273
#define DPROPREF 274
#define DCLASS 275
#define DCLASSDEF 276
#define DCTOR 277
#define DENDCLASS 278
#define DFROM 279
#define DEXTERN 280
#define DMODULE 281
#define DLOAD 282
#define DIMPORT 283
#define DALIAS 284
#define DSWITCH 285
#define DSELECT 286
#define DCASE 287
#define DENDSWITCH 288
#define DLINE 289
#define DSTRING 290
#define DISTRING 291
#define DCSTRING 292
#define DHAS 293
#define DHASNT 294
#define DINHERIT 295
#define DINSTANCE 296
#define SYMBOL 297
#define EXPORT 298
#define LABEL 299
#define INTEGER 300
#define REG_A 301
#define REG_B 302
#define REG_S1 303
#define REG_S2 304
#define REG_L1 305
#define REG_L2 306
#define NUMERIC 307
#define STRING 308
#define STRING_ID 309
#define TRUE_TOKEN 310
#define FALSE_TOKEN 311
#define I_LD 312
#define I_LNIL 313
#define NIL 314
#define I_ADD 315
#define I_SUB 316
#define I_MUL 317
#define I_DIV 318
#define I_MOD 319
#define I_POW 320
#define I_ADDS 321
#define I_SUBS 322
#define I_MULS 323
#define I_DIVS 324
#define I_POWS 325
#define I_INC 326
#define I_DEC 327
#define I_INCP 328
#define I_DECP 329
#define I_NEG 330
#define I_NOT 331
#define I_RET 332
#define I_RETV 333
#define I_RETA 334
#define I_FORK 335
#define I_PUSH 336
#define I_PSHR 337
#define I_PSHN 338
#define I_POP 339
#define I_LDV 340
#define I_LDVT 341
#define I_STV 342
#define I_STVR 343
#define I_STVS 344
#define I_LDP 345
#define I_LDPT 346
#define I_STP 347
#define I_STPR 348
#define I_STPS 349
#define I_TRAV 350
#define I_TRAN 351
#define I_TRAL 352
#define I_IPOP 353
#define I_XPOP 354
#define I_GENA 355
#define I_GEND 356
#define I_GENR 357
#define I_GEOR 358
#define I_JMP 359
#define I_IFT 360
#define I_IFF 361
#define I_BOOL 362
#define I_EQ 363
#define I_NEQ 364
#define I_GT 365
#define I_GE 366
#define I_LT 367
#define I_LE 368
#define I_UNPK 369
#define I_UNPS 370
#define I_CALL 371
#define I_INST 372
#define I_SWCH 373
#define I_SELE 374
#define I_NOP 375
#define I_TRY 376
#define I_JTRY 377
#define I_PTRY 378
#define I_RIS 379
#define I_LDRF 380
#define I_ONCE 381
#define I_BAND 382
#define I_BOR 383
#define I_BXOR 384
#define I_BNOT 385
#define I_MODS 386
#define I_AND 387
#define I_OR 388
#define I_ANDS 389
#define I_ORS 390
#define I_XORS 391
#define I_NOTS 392
#define I_HAS 393
#define I_HASN 394
#define I_GIVE 395
#define I_GIVN 396
#define I_IN 397
#define I_NOIN 398
#define I_PROV 399
#define I_END 400
#define I_PEEK 401
#define I_PSIN 402
#define I_PASS 403
#define I_SHR 404
#define I_SHL 405
#define I_SHRS 406
#define I_SHLS 407
#define I_LDVR 408
#define I_LDPR 409
#define I_LSB 410
#define I_INDI 411
#define I_STEX 412
#define I_TRAC 413
#define I_WRT 414
#define I_STO 415
#define I_FORB 416
#define I_EVAL 417
#define I_OOB 418




/* Copy the first part of user declarations.  */
#line 17 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"

#include <falcon/setup.h>
#include <stdio.h>
#include <iostream>
#include <ctype.h>

#include <fasm/comp.h>
#include <fasm/clexer.h>
#include <fasm/pseudo.h>
#include <falcon/string.h>
#include <falcon/pcodes.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::AsmCompiler *>(fasm_param) )
#define  LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM fasm_param
#define YYLEX_PARAM fasm_param
#define YYSTYPE Falcon::Pseudo *

int fasm_parse( void *param );
void fasm_error( const char *s );

inline int yylex (void *lvalp, void *fasm_param)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 478 "/home/gian/Progetti/falcon/core/engine/fasm_parser.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1910

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  164
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  124
/* YYNRULES -- Number of rules.  */
#define YYNRULES  454
/* YYNRULES -- Number of states.  */
#define YYNSTATES  824

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   418

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,     9,    12,    16,    19,    22,
      25,    27,    29,    31,    33,    35,    37,    39,    41,    43,
      45,    47,    49,    51,    53,    55,    57,    59,    61,    63,
      65,    67,    69,    71,    73,    75,    77,    80,    84,    89,
      94,   100,   104,   109,   113,   118,   122,   126,   129,   133,
     137,   142,   144,   147,   150,   154,   159,   164,   170,   176,
     182,   188,   193,   198,   203,   208,   213,   218,   223,   228,
     233,   240,   242,   246,   250,   254,   257,   260,   265,   271,
     275,   280,   283,   287,   290,   292,   295,   298,   302,   305,
     308,   311,   314,   316,   320,   322,   326,   329,   330,   332,
     334,   336,   338,   340,   342,   344,   346,   348,   350,   352,
     354,   356,   358,   360,   362,   364,   366,   368,   370,   372,
     374,   376,   378,   380,   382,   384,   386,   388,   390,   392,
     394,   396,   398,   400,   402,   404,   406,   408,   410,   412,
     414,   416,   418,   420,   422,   424,   426,   428,   430,   432,
     434,   436,   438,   440,   442,   444,   446,   448,   450,   452,
     454,   456,   458,   460,   462,   464,   466,   468,   470,   472,
     474,   476,   478,   480,   482,   484,   486,   488,   490,   492,
     494,   496,   498,   500,   502,   504,   506,   508,   510,   512,
     514,   516,   518,   520,   522,   524,   526,   528,   530,   532,
     534,   536,   538,   540,   542,   547,   550,   555,   558,   561,
     564,   569,   572,   577,   580,   585,   588,   593,   596,   601,
     604,   609,   612,   617,   620,   625,   628,   633,   636,   641,
     644,   649,   652,   657,   660,   665,   668,   673,   676,   681,
     684,   689,   692,   695,   698,   701,   704,   707,   710,   713,
     716,   719,   722,   725,   728,   731,   734,   737,   742,   747,
     750,   755,   760,   763,   768,   771,   776,   779,   782,   784,
     787,   790,   793,   796,   799,   802,   805,   808,   811,   816,
     821,   824,   831,   838,   841,   848,   855,   858,   865,   872,
     875,   880,   885,   888,   893,   898,   901,   908,   915,   918,
     925,   932,   935,   942,   949,   952,   957,   962,   965,   972,
     979,   982,   989,   996,   999,  1002,  1005,  1008,  1011,  1014,
    1017,  1020,  1023,  1026,  1033,  1036,  1039,  1042,  1045,  1048,
    1051,  1054,  1057,  1060,  1063,  1068,  1073,  1076,  1081,  1086,
    1089,  1094,  1099,  1102,  1105,  1108,  1111,  1113,  1116,  1118,
    1121,  1124,  1127,  1129,  1132,  1135,  1138,  1140,  1143,  1150,
    1153,  1160,  1163,  1167,  1173,  1178,  1183,  1186,  1191,  1196,
    1201,  1206,  1209,  1214,  1219,  1224,  1229,  1232,  1237,  1242,
    1247,  1252,  1255,  1258,  1261,  1264,  1269,  1272,  1277,  1280,
    1285,  1288,  1293,  1296,  1301,  1304,  1309,  1312,  1317,  1320,
    1323,  1326,  1331,  1336,  1339,  1344,  1349,  1352,  1357,  1362,
    1365,  1370,  1375,  1378,  1383,  1386,  1391,  1394,  1399,  1402,
    1405,  1408,  1411,  1414,  1419,  1422,  1427,  1430,  1435,  1438,
    1443,  1446,  1451,  1454,  1459,  1462,  1467,  1470,  1473,  1476,
    1479,  1482,  1485,  1488,  1491,  1494,  1497,  1500,  1505,  1508,
    1513,  1516,  1519,  1522,  1527
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     165,     0,    -1,    -1,   165,   166,    -1,     3,    -1,   176,
       3,    -1,   179,   181,     3,    -1,   179,     3,    -1,   181,
       3,    -1,     1,     3,    -1,    59,    -1,   168,    -1,   169,
      -1,   172,    -1,    42,    -1,   170,    -1,    46,    -1,    47,
      -1,    48,    -1,    49,    -1,    50,    -1,    51,    -1,    59,
      -1,   172,    -1,    52,    -1,    55,    -1,    56,    -1,   173,
      -1,    54,    -1,    53,    -1,    45,    -1,    54,    -1,    53,
      -1,    53,    -1,     4,    -1,     7,    -1,    26,     4,    -1,
       8,     4,   180,    -1,     8,     4,   180,    43,    -1,     9,
       4,   171,   180,    -1,     9,     4,   171,   180,    43,    -1,
      10,     4,   171,    -1,    10,     4,   171,    43,    -1,    11,
       4,   180,    -1,    11,     4,   180,    43,    -1,    12,     4,
     180,    -1,    13,     4,   180,    -1,    14,     4,    -1,    14,
       4,    43,    -1,    16,     4,   180,    -1,    16,     4,   180,
      43,    -1,    15,    -1,    27,     4,    -1,    27,    53,    -1,
      28,     4,   180,    -1,    28,     4,   180,     4,    -1,    28,
       4,   180,    53,    -1,    28,     4,   180,     4,   175,    -1,
      28,     4,   180,    53,   175,    -1,    29,     4,   180,    53,
     175,    -1,    29,     4,   180,     4,   175,    -1,    30,   169,
       5,     4,    -1,    30,   169,     5,    45,    -1,    31,   169,
       5,     4,    -1,    31,   169,     5,    45,    -1,    32,    59,
       5,     4,    -1,    32,    45,     5,     4,    -1,    32,    53,
       5,     4,    -1,    32,    54,     5,     4,    -1,    32,    42,
       5,     4,    -1,    32,    45,     6,    45,     5,     4,    -1,
      33,    -1,    18,     4,   171,    -1,    18,     4,    42,    -1,
      19,     4,    42,    -1,    38,   177,    -1,    39,   178,    -1,
      41,    42,     4,   180,    -1,    41,    42,     4,   180,    43,
      -1,    20,     4,   180,    -1,    20,     4,   180,    43,    -1,
      21,     4,    -1,    21,     4,    43,    -1,    22,    42,    -1,
      23,    -1,    40,    42,    -1,    24,     4,    -1,    25,     4,
     180,    -1,    34,    45,    -1,    35,    53,    -1,    36,    53,
      -1,    37,    53,    -1,    42,    -1,   177,     5,    42,    -1,
      42,    -1,   178,     5,    42,    -1,    44,     6,    -1,    -1,
      45,    -1,   182,    -1,   184,    -1,   185,    -1,   187,    -1,
     189,    -1,   191,    -1,   193,    -1,   194,    -1,   186,    -1,
     188,    -1,   190,    -1,   192,    -1,   202,    -1,   203,    -1,
     204,    -1,   205,    -1,   195,    -1,   196,    -1,   197,    -1,
     198,    -1,   199,    -1,   200,    -1,   206,    -1,   207,    -1,
     212,    -1,   213,    -1,   214,    -1,   216,    -1,   217,    -1,
     218,    -1,   219,    -1,   220,    -1,   221,    -1,   222,    -1,
     223,    -1,   224,    -1,   226,    -1,   225,    -1,   227,    -1,
     228,    -1,   229,    -1,   230,    -1,   231,    -1,   232,    -1,
     233,    -1,   234,    -1,   236,    -1,   237,    -1,   238,    -1,
     239,    -1,   240,    -1,   210,    -1,   211,    -1,   208,    -1,
     209,    -1,   242,    -1,   244,    -1,   243,    -1,   245,    -1,
     201,    -1,   246,    -1,   241,    -1,   235,    -1,   248,    -1,
     249,    -1,   183,    -1,   247,    -1,   252,    -1,   253,    -1,
     254,    -1,   255,    -1,   256,    -1,   257,    -1,   258,    -1,
     259,    -1,   260,    -1,   263,    -1,   261,    -1,   262,    -1,
     264,    -1,   265,    -1,   266,    -1,   267,    -1,   268,    -1,
     269,    -1,   270,    -1,   251,    -1,   215,    -1,   271,    -1,
     272,    -1,   273,    -1,   274,    -1,   275,    -1,   276,    -1,
     277,    -1,   278,    -1,   279,    -1,   280,    -1,   281,    -1,
     282,    -1,   283,    -1,   284,    -1,   285,    -1,   286,    -1,
     287,    -1,    57,   169,     5,   167,    -1,    57,     1,    -1,
     125,   169,     5,   167,    -1,   125,     1,    -1,    58,   169,
      -1,    58,     1,    -1,    60,   167,     5,   167,    -1,    60,
       1,    -1,    66,   169,     5,   167,    -1,    66,     1,    -1,
      61,   167,     5,   167,    -1,    61,     1,    -1,    67,   169,
       5,   167,    -1,    67,     1,    -1,    62,   167,     5,   167,
      -1,    62,     1,    -1,    68,   169,     5,   167,    -1,    68,
       1,    -1,    63,   167,     5,   167,    -1,    63,     1,    -1,
      69,   169,     5,   167,    -1,    69,     1,    -1,    64,   167,
       5,   167,    -1,    64,     1,    -1,    65,   167,     5,   167,
      -1,    65,     1,    -1,   108,   167,     5,   167,    -1,   108,
       1,    -1,   109,   167,     5,   167,    -1,   109,     1,    -1,
     111,   167,     5,   167,    -1,   111,     1,    -1,   110,   167,
       5,   167,    -1,   110,     1,    -1,   113,   167,     5,   167,
      -1,   113,     1,    -1,   112,   167,     5,   167,    -1,   112,
       1,    -1,   121,     4,    -1,   121,    45,    -1,   121,     1,
      -1,    71,   169,    -1,    71,     1,    -1,    72,   169,    -1,
      72,     1,    -1,    73,   169,    -1,    73,     1,    -1,    74,
     169,    -1,    74,     1,    -1,    75,   167,    -1,    75,     1,
      -1,    76,   167,    -1,    76,     1,    -1,   116,    45,     5,
     169,    -1,   116,    45,     5,     4,    -1,   116,     1,    -1,
     117,    45,     5,   169,    -1,   117,    45,     5,     4,    -1,
     117,     1,    -1,   114,   169,     5,   169,    -1,   114,     1,
      -1,   115,    45,     5,   169,    -1,   115,     1,    -1,    81,
     167,    -1,    83,    -1,    81,     1,    -1,    82,   167,    -1,
      82,     1,    -1,    84,   169,    -1,    84,     1,    -1,   146,
     169,    -1,   146,     1,    -1,    99,   169,    -1,    99,     1,
      -1,    85,   169,     5,   169,    -1,    85,   169,     5,   173,
      -1,    85,     1,    -1,    86,   169,     5,   169,     5,   169,
      -1,    86,   169,     5,   173,     5,   169,    -1,    86,     1,
      -1,    87,   169,     5,   169,     5,   167,    -1,    87,   169,
       5,   173,     5,   167,    -1,    87,     1,    -1,    88,   169,
       5,   169,     5,   167,    -1,    88,   169,     5,   173,     5,
     167,    -1,    88,     1,    -1,    89,   169,     5,   169,    -1,
      89,   169,     5,   173,    -1,    89,     1,    -1,    90,   169,
       5,   169,    -1,    90,   169,     5,   173,    -1,    90,     1,
      -1,    91,   169,     5,   169,     5,   169,    -1,    91,   169,
       5,   173,     5,   169,    -1,    91,     1,    -1,    92,   169,
       5,   169,     5,   167,    -1,    92,   169,     5,   173,     5,
     167,    -1,    92,     1,    -1,    93,   169,     5,   169,     5,
     169,    -1,    93,   169,     5,   173,     5,   169,    -1,    93,
       1,    -1,    94,   169,     5,   169,    -1,    94,   169,     5,
     173,    -1,    94,     1,    -1,    95,    45,     5,   167,     5,
     167,    -1,    95,     4,     5,   167,     5,   167,    -1,    95,
       1,    -1,    96,     4,     5,     4,     5,    45,    -1,    96,
      45,     5,    45,     5,    45,    -1,    96,     1,    -1,    97,
      45,    -1,    97,     4,    -1,    97,     1,    -1,    98,    45,
      -1,    98,     1,    -1,   100,    45,    -1,   100,     1,    -1,
     101,    45,    -1,   101,     1,    -1,   102,   168,     5,   168,
       5,   171,    -1,   102,     1,    -1,   103,   168,    -1,   103,
       1,    -1,   124,   167,    -1,   124,     1,    -1,   104,     4,
      -1,   104,    45,    -1,   104,     1,    -1,   107,   167,    -1,
     107,     1,    -1,   105,     4,     5,   167,    -1,   105,    45,
       5,   167,    -1,   105,     1,    -1,   106,     4,     5,   167,
      -1,   106,    45,     5,   167,    -1,   106,     1,    -1,    80,
      45,     5,     4,    -1,    80,    45,     5,    45,    -1,    80,
       1,    -1,   122,     4,    -1,   122,    45,    -1,   122,     1,
      -1,    77,    -1,    77,     1,    -1,    79,    -1,    79,     1,
      -1,    78,   167,    -1,    78,     1,    -1,   120,    -1,   120,
       1,    -1,   123,    45,    -1,   123,     1,    -1,   145,    -1,
     145,     1,    -1,   118,    45,     5,   169,     5,   250,    -1,
     118,     1,    -1,   119,    45,     5,   169,     5,   250,    -1,
     119,     1,    -1,    45,     5,     4,    -1,   250,     5,    45,
       5,     4,    -1,   126,    45,     5,   167,    -1,   126,     4,
       5,   167,    -1,   126,     1,    -1,   127,    42,     5,    42,
      -1,   127,    42,     5,    45,    -1,   127,    45,     5,    42,
      -1,   127,    45,     5,    45,    -1,   127,     1,    -1,   128,
      42,     5,    42,    -1,   128,    42,     5,    45,    -1,   128,
      45,     5,    42,    -1,   128,    45,     5,    45,    -1,   128,
       1,    -1,   129,    42,     5,    42,    -1,   129,    42,     5,
      45,    -1,   129,    45,     5,    42,    -1,   129,    45,     5,
      45,    -1,   129,     1,    -1,   130,    42,    -1,   130,    45,
      -1,   130,     1,    -1,   132,   167,     5,   167,    -1,   132,
       1,    -1,   133,   167,     5,   167,    -1,   133,     1,    -1,
     134,   169,     5,   168,    -1,   134,     1,    -1,   135,   169,
       5,   168,    -1,   135,     1,    -1,   136,   169,     5,   168,
      -1,   136,     1,    -1,   131,   169,     5,   168,    -1,   131,
       1,    -1,    70,   169,     5,   168,    -1,    70,     1,    -1,
     137,   169,    -1,   137,     1,    -1,   138,   169,     5,   169,
      -1,   138,   169,     5,    45,    -1,   138,     1,    -1,   139,
     169,     5,   169,    -1,   139,   169,     5,    45,    -1,   139,
       1,    -1,   140,   169,     5,   169,    -1,   140,   169,     5,
      45,    -1,   140,     1,    -1,   141,   169,     5,   169,    -1,
     141,   169,     5,    45,    -1,   141,     1,    -1,   142,   167,
       5,   168,    -1,   142,     1,    -1,   143,   167,     5,   168,
      -1,   143,     1,    -1,   144,   167,     5,   168,    -1,   144,
       1,    -1,   147,   169,    -1,   147,     1,    -1,   148,   169,
      -1,   148,     1,    -1,   149,   167,     5,   167,    -1,   149,
       1,    -1,   150,   167,     5,   167,    -1,   150,     1,    -1,
     151,   167,     5,   167,    -1,   151,     1,    -1,   152,   167,
       5,   167,    -1,   152,     1,    -1,   153,   167,     5,   167,
      -1,   153,     1,    -1,   154,   167,     5,   167,    -1,   154,
       1,    -1,   155,   167,     5,   167,    -1,   155,     1,    -1,
     156,   174,    -1,   156,   169,    -1,   156,     1,    -1,   157,
     174,    -1,   157,   169,    -1,   157,     1,    -1,   158,   167,
      -1,   158,     1,    -1,   159,   167,    -1,   159,     1,    -1,
     160,   169,     5,   167,    -1,   160,     1,    -1,   161,   174,
       5,   167,    -1,   161,     1,    -1,   162,   167,    -1,   162,
       1,    -1,   163,    45,     5,   167,    -1,   163,     1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   234,   234,   236,   240,   241,   242,   243,   244,   245,
     248,   248,   249,   249,   250,   250,   251,   251,   251,   251,
     251,   251,   253,   253,   254,   254,   254,   254,   255,   255,
     255,   256,   256,   257,   257,   261,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   286,   289,
     292,   295,   296,   297,   298,   299,   300,   301,   302,   303,
     304,   305,   306,   307,   308,   309,   310,   311,   312,   313,
     314,   315,   316,   317,   318,   319,   320,   321,   322,   323,
     328,   334,   343,   344,   348,   349,   352,   356,   357,   361,
     362,   363,   364,   365,   366,   367,   368,   369,   370,   371,
     372,   373,   374,   375,   376,   377,   378,   379,   380,   381,
     382,   383,   384,   385,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,   406,   407,   408,   409,   410,   411,
     412,   413,   414,   415,   416,   417,   418,   419,   420,   421,
     422,   423,   424,   425,   426,   427,   428,   429,   430,   431,
     432,   433,   434,   435,   436,   437,   438,   439,   440,   441,
     442,   443,   444,   445,   446,   447,   448,   449,   450,   451,
     452,   453,   454,   455,   456,   457,   458,   459,   460,   461,
     462,   463,   464,   465,   469,   470,   474,   475,   479,   480,
     484,   485,   489,   490,   495,   496,   500,   501,   505,   506,
     510,   511,   516,   517,   521,   522,   526,   527,   531,   532,
     537,   538,   542,   543,   547,   548,   552,   553,   557,   558,
     562,   563,   567,   568,   569,   573,   574,   578,   579,   584,
     585,   589,   590,   595,   596,   600,   601,   605,   606,   607,
     611,   612,   613,   617,   618,   622,   623,   628,   629,   630,
     634,   635,   640,   641,   645,   646,   650,   651,   656,   657,
     658,   662,   663,   664,   668,   669,   670,   674,   675,   676,
     680,   681,   682,   686,   687,   688,   692,   693,   694,   698,
     699,   700,   704,   705,   706,   710,   711,   712,   716,   717,
     718,   722,   723,   724,   728,   729,   730,   734,   735,   739,
     740,   744,   745,   749,   750,   754,   755,   759,   760,   764,
     765,   766,   770,   771,   775,   776,   777,   781,   782,   783,
     788,   789,   790,   794,   795,   796,   800,   801,   805,   806,
     810,   811,   815,   816,   820,   821,   825,   826,   830,   831,
     835,   836,   840,   849,   858,   859,   860,   864,   865,   866,
     867,   868,   872,   873,   874,   875,   876,   880,   881,   882,
     883,   884,   888,   889,   890,   894,   895,   899,   900,   904,
     905,   909,   910,   914,   915,   919,   920,   924,   925,   929,
     930,   934,   935,   936,   940,   941,   942,   946,   947,   948,
     952,   953,   954,   959,   960,   964,   965,   969,   970,   974,
     975,   979,   980,   984,   985,   989,   990,   994,   995,   999,
    1000,  1004,  1005,  1009,  1010,  1014,  1015,  1019,  1020,  1021,
    1025,  1026,  1027,  1031,  1032,  1036,  1037,  1042,  1043,  1047,
    1048,  1052,  1053,  1057,  1058
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "NAME", "COMMA", "COLON",
  "DENTRY", "DGLOBAL", "DVAR", "DCONST", "DATTRIB", "DLOCAL", "DPARAM",
  "DFUNCDEF", "DENDFUNC", "DFUNC", "DMETHOD", "DPROP", "DPROPREF",
  "DCLASS", "DCLASSDEF", "DCTOR", "DENDCLASS", "DFROM", "DEXTERN",
  "DMODULE", "DLOAD", "DIMPORT", "DALIAS", "DSWITCH", "DSELECT", "DCASE",
  "DENDSWITCH", "DLINE", "DSTRING", "DISTRING", "DCSTRING", "DHAS",
  "DHASNT", "DINHERIT", "DINSTANCE", "SYMBOL", "EXPORT", "LABEL",
  "INTEGER", "REG_A", "REG_B", "REG_S1", "REG_S2", "REG_L1", "REG_L2",
  "NUMERIC", "STRING", "STRING_ID", "TRUE_TOKEN", "FALSE_TOKEN", "I_LD",
  "I_LNIL", "NIL", "I_ADD", "I_SUB", "I_MUL", "I_DIV", "I_MOD", "I_POW",
  "I_ADDS", "I_SUBS", "I_MULS", "I_DIVS", "I_POWS", "I_INC", "I_DEC",
  "I_INCP", "I_DECP", "I_NEG", "I_NOT", "I_RET", "I_RETV", "I_RETA",
  "I_FORK", "I_PUSH", "I_PSHR", "I_PSHN", "I_POP", "I_LDV", "I_LDVT",
  "I_STV", "I_STVR", "I_STVS", "I_LDP", "I_LDPT", "I_STP", "I_STPR",
  "I_STPS", "I_TRAV", "I_TRAN", "I_TRAL", "I_IPOP", "I_XPOP", "I_GENA",
  "I_GEND", "I_GENR", "I_GEOR", "I_JMP", "I_IFT", "I_IFF", "I_BOOL",
  "I_EQ", "I_NEQ", "I_GT", "I_GE", "I_LT", "I_LE", "I_UNPK", "I_UNPS",
  "I_CALL", "I_INST", "I_SWCH", "I_SELE", "I_NOP", "I_TRY", "I_JTRY",
  "I_PTRY", "I_RIS", "I_LDRF", "I_ONCE", "I_BAND", "I_BOR", "I_BXOR",
  "I_BNOT", "I_MODS", "I_AND", "I_OR", "I_ANDS", "I_ORS", "I_XORS",
  "I_NOTS", "I_HAS", "I_HASN", "I_GIVE", "I_GIVN", "I_IN", "I_NOIN",
  "I_PROV", "I_END", "I_PEEK", "I_PSIN", "I_PASS", "I_SHR", "I_SHL",
  "I_SHRS", "I_SHLS", "I_LDVR", "I_LDPR", "I_LSB", "I_INDI", "I_STEX",
  "I_TRAC", "I_WRT", "I_STO", "I_FORB", "I_EVAL", "I_OOB", "$accept",
  "input", "line", "xoperand", "operand", "op_variable", "op_register",
  "x_op_immediate", "op_immediate", "op_scalar", "op_string",
  "string_or_name", "directive", "has_symlist", "hasnt_symlist", "label",
  "def_line", "instruction", "inst_ld", "inst_ldrf", "inst_ldnil",
  "inst_add", "inst_adds", "inst_sub", "inst_subs", "inst_mul",
  "inst_muls", "inst_div", "inst_divs", "inst_mod", "inst_pow", "inst_eq",
  "inst_ne", "inst_ge", "inst_gt", "inst_le", "inst_lt", "inst_try",
  "inst_inc", "inst_dec", "inst_incp", "inst_decp", "inst_neg", "inst_not",
  "inst_call", "inst_inst", "inst_unpk", "inst_unps", "inst_push",
  "inst_pshr", "inst_pop", "inst_peek", "inst_xpop", "inst_ldv",
  "inst_ldvt", "inst_stv", "inst_stvr", "inst_stvs", "inst_ldp",
  "inst_ldpt", "inst_stp", "inst_stpr", "inst_stps", "inst_trav",
  "inst_tran", "inst_tral", "inst_ipop", "inst_gena", "inst_gend",
  "inst_genr", "inst_geor", "inst_ris", "inst_jmp", "inst_bool",
  "inst_ift", "inst_iff", "inst_fork", "inst_jtry", "inst_ret",
  "inst_reta", "inst_retval", "inst_nop", "inst_ptry", "inst_end",
  "inst_swch", "inst_sele", "switch_list", "inst_once", "inst_band",
  "inst_bor", "inst_bxor", "inst_bnot", "inst_and", "inst_or", "inst_ands",
  "inst_ors", "inst_xors", "inst_mods", "inst_pows", "inst_nots",
  "inst_has", "inst_hasn", "inst_give", "inst_givn", "inst_in",
  "inst_noin", "inst_prov", "inst_psin", "inst_pass", "inst_shr",
  "inst_shl", "inst_shrs", "inst_shls", "inst_ldvr", "inst_ldpr",
  "inst_lsb", "inst_indi", "inst_stex", "inst_trac", "inst_wrt",
  "inst_sto", "inst_forb", "inst_eval", "inst_oob", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414,
     415,   416,   417,   418
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   164,   165,   165,   166,   166,   166,   166,   166,   166,
     167,   167,   168,   168,   169,   169,   170,   170,   170,   170,
     170,   170,   171,   171,   172,   172,   172,   172,   173,   173,
     173,   174,   174,   175,   175,   176,   176,   176,   176,   176,
     176,   176,   176,   176,   176,   176,   176,   176,   176,   176,
     176,   176,   176,   176,   176,   176,   176,   176,   176,   176,
     176,   176,   176,   176,   176,   176,   176,   176,   176,   176,
     176,   176,   176,   176,   176,   176,   176,   176,   176,   176,
     176,   176,   176,   176,   176,   176,   176,   176,   176,   176,
     176,   176,   177,   177,   178,   178,   179,   180,   180,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   182,   182,   183,   183,   184,   184,
     185,   185,   186,   186,   187,   187,   188,   188,   189,   189,
     190,   190,   191,   191,   192,   192,   193,   193,   194,   194,
     195,   195,   196,   196,   197,   197,   198,   198,   199,   199,
     200,   200,   201,   201,   201,   202,   202,   203,   203,   204,
     204,   205,   205,   206,   206,   207,   207,   208,   208,   208,
     209,   209,   209,   210,   210,   211,   211,   212,   212,   212,
     213,   213,   214,   214,   215,   215,   216,   216,   217,   217,
     217,   218,   218,   218,   219,   219,   219,   220,   220,   220,
     221,   221,   221,   222,   222,   222,   223,   223,   223,   224,
     224,   224,   225,   225,   225,   226,   226,   226,   227,   227,
     227,   228,   228,   228,   229,   229,   229,   230,   230,   231,
     231,   232,   232,   233,   233,   234,   234,   235,   235,   236,
     236,   236,   237,   237,   238,   238,   238,   239,   239,   239,
     240,   240,   240,   241,   241,   241,   242,   242,   243,   243,
     244,   244,   245,   245,   246,   246,   247,   247,   248,   248,
     249,   249,   250,   250,   251,   251,   251,   252,   252,   252,
     252,   252,   253,   253,   253,   253,   253,   254,   254,   254,
     254,   254,   255,   255,   255,   256,   256,   257,   257,   258,
     258,   259,   259,   260,   260,   261,   261,   262,   262,   263,
     263,   264,   264,   264,   265,   265,   265,   266,   266,   266,
     267,   267,   267,   268,   268,   269,   269,   270,   270,   271,
     271,   272,   272,   273,   273,   274,   274,   275,   275,   276,
     276,   277,   277,   278,   278,   279,   279,   280,   280,   280,
     281,   281,   281,   282,   282,   283,   283,   284,   284,   285,
     285,   286,   286,   287,   287
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     1,     2,     3,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     3,     4,     4,
       5,     3,     4,     3,     4,     3,     3,     2,     3,     3,
       4,     1,     2,     2,     3,     4,     4,     5,     5,     5,
       5,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       6,     1,     3,     3,     3,     2,     2,     4,     5,     3,
       4,     2,     3,     2,     1,     2,     2,     3,     2,     2,
       2,     2,     1,     3,     1,     3,     2,     0,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     4,     2,     4,     2,     2,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     4,     4,     2,
       4,     4,     2,     4,     2,     4,     2,     2,     1,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     4,     4,
       2,     6,     6,     2,     6,     6,     2,     6,     6,     2,
       4,     4,     2,     4,     4,     2,     6,     6,     2,     6,
       6,     2,     6,     6,     2,     4,     4,     2,     6,     6,
       2,     6,     6,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     6,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     4,     4,     2,     4,     4,     2,
       4,     4,     2,     2,     2,     2,     1,     2,     1,     2,
       2,     2,     1,     2,     2,     2,     1,     2,     6,     2,
       6,     2,     3,     5,     4,     4,     2,     4,     4,     4,
       4,     2,     4,     4,     4,     4,     2,     4,     4,     4,
       4,     2,     2,     2,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     2,
       2,     4,     4,     2,     4,     4,     2,     4,     4,     2,
       4,     4,     2,     4,     2,     4,     2,     4,     2,     2,
       2,     2,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     4,     2,     4,
       2,     2,     2,     4,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,     0,     4,    35,     0,     0,     0,     0,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
      84,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      71,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   268,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     3,     0,     0,     0,
      99,   164,   100,   101,   107,   102,   108,   103,   109,   104,
     110,   105,   106,   115,   116,   117,   118,   119,   120,   158,
     111,   112,   113,   114,   121,   122,   152,   153,   150,   151,
     123,   124,   125,   186,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   136,   135,   137,   138,   139,   140,   141,
     142,   143,   144,   161,   145,   146,   147,   148,   149,   160,
     154,   156,   155,   157,   159,   165,   162,   163,   185,   166,
     167,   168,   169,   170,   171,   172,   173,   174,   176,   177,
     175,   178,   179,   180,   181,   182,   183,   184,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   202,   203,     9,    97,     0,     0,    97,
      97,    97,    47,    97,     0,     0,    97,    81,    83,    86,
      97,    36,    52,    53,    97,    97,    14,    16,    17,    18,
      19,    20,    21,     0,    15,     0,     0,     0,     0,     0,
       0,    88,    89,    90,    91,    92,    75,    94,    76,    85,
       0,    96,   205,     0,   209,   208,   211,    30,    24,    29,
      28,    25,    26,    10,     0,    11,    12,    13,    27,   215,
       0,   219,     0,   223,     0,   227,     0,   229,     0,   213,
       0,   217,     0,   221,     0,   225,     0,   398,     0,   246,
     245,   248,   247,   250,   249,   252,   251,   254,   253,   256,
     255,   347,   351,   350,   349,   342,     0,   269,   267,   271,
     270,   273,   272,   280,     0,   283,     0,   286,     0,   289,
       0,   292,     0,   295,     0,   298,     0,   301,     0,   304,
       0,   307,     0,   310,     0,     0,   313,     0,     0,   316,
     315,   314,   318,   317,   277,   276,   320,   319,   322,   321,
     324,     0,   326,   325,   331,   329,   330,   336,     0,     0,
     339,     0,     0,   333,   332,   231,     0,   233,     0,   237,
       0,   235,     0,   241,     0,   239,     0,   264,     0,   266,
       0,   259,     0,   262,     0,   359,     0,   361,     0,   353,
     244,   242,   243,   345,   343,   344,   355,   354,   328,   327,
     207,     0,   366,     0,     0,   371,     0,     0,   376,     0,
       0,   381,     0,     0,   384,   382,   383,   396,     0,   386,
       0,   388,     0,   390,     0,   392,     0,   394,     0,   400,
     399,   403,     0,   406,     0,   409,     0,   412,     0,   414,
       0,   416,     0,   418,     0,   357,   275,   274,   420,   419,
     422,   421,   424,     0,   426,     0,   428,     0,   430,     0,
     432,     0,   434,     0,   436,     0,   439,    32,    31,   438,
     437,   442,   441,   440,   444,   443,   446,   445,   448,     0,
     450,     0,   452,   451,   454,     0,     5,     7,     0,     8,
      98,    37,    22,    97,    23,    41,    43,    45,    46,    48,
      49,    73,    72,    74,    79,    82,    87,    54,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    97,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     6,    38,    39,    42,
      44,    50,    80,    55,    56,     0,     0,    61,    62,    63,
      64,    69,    66,     0,    67,    68,    65,    93,    95,    77,
     204,   210,   214,   218,   222,   226,   228,   212,   216,   220,
     224,   397,   340,   341,   278,   279,     0,     0,     0,     0,
       0,     0,   290,   291,   293,   294,     0,     0,     0,     0,
       0,     0,   305,   306,     0,     0,     0,     0,     0,   334,
     335,   337,   338,   230,   232,   236,   234,   240,   238,   263,
     265,   258,   257,   261,   260,     0,     0,   206,   365,   364,
     367,   368,   369,   370,   372,   373,   374,   375,   377,   378,
     379,   380,   395,   385,   387,   389,   391,   393,   402,   401,
     405,   404,   408,   407,   411,   410,   413,   415,   417,   423,
     425,   427,   429,   431,   433,   435,   447,   449,   453,    40,
      34,    33,    57,    58,    60,    59,     0,    78,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    70,   281,   282,
     284,   285,   287,   288,   296,   297,   299,   300,   302,   303,
     309,   308,   311,   312,   323,     0,   358,   360,     0,     0,
     362,     0,     0,   363
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,   146,   314,   315,   316,   284,   543,   317,   318,
     520,   772,   147,   296,   298,   148,   541,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   177,   178,   179,   180,   181,
     182,   183,   184,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   816,   218,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -478
static const yytype_int16 yypact[] =
{
    -478,   318,  -478,     8,  -478,  -478,    14,    31,    59,    69,
      79,   105,   137,  -478,   144,   155,   156,   168,   177,   157,
    -478,   196,   306,   307,     4,   308,   309,   470,   470,   701,
    -478,   269,   262,   264,   271,   323,   324,   331,   332,   371,
      96,   321,    81,   565,   668,   683,   725,   740,   600,   782,
    1339,  1356,  1366,  1381,  1391,  1408,  1418,   755,   811,   119,
     826,   181,    36,   844,   868,  -478,  1433,  1443,  1460,  1470,
    1485,  1495,  1512,  1522,  1537,  1547,  1564,    16,   113,   319,
      55,  1574,   315,   510,   481,  1324,   493,   494,   503,   883,
     901,   925,   940,   958,   982,   997,  1589,   523,   524,   542,
     577,   578,   187,   504,   505,   608,  1015,  1599,   509,   138,
     628,   629,   651,  1616,  1039,  1054,  1626,  1641,  1651,  1668,
    1678,  1693,  1703,  1720,  1072,  1096,  1111,   188,  1730,  1745,
    1755,  1129,  1153,  1168,  1186,  1210,  1225,  1243,   115,   546,
    1267,  1282,  1772,     6,  1300,   627,  -478,   490,   146,   497,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,   451,  1824,  1824,   451,
     451,   451,   458,   451,  1797,   460,   451,   471,  -478,  -478,
     451,  -478,  -478,  -478,   451,   451,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,   498,  -478,   517,   572,    93,   575,   584,
     585,  -478,  -478,  -478,  -478,  -478,   586,  -478,   593,  -478,
     602,  -478,  -478,   603,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,   620,  -478,  -478,  -478,  -478,  -478,
     621,  -478,   622,  -478,   638,  -478,   639,  -478,   652,  -478,
     653,  -478,   673,  -478,   674,  -478,   675,  -478,   676,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,   677,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,   689,  -478,   690,  -478,   702,  -478,
     703,  -478,   704,  -478,   706,  -478,   707,  -478,   735,  -478,
     739,  -478,   742,  -478,   743,   744,  -478,   745,   748,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,   754,  -478,  -478,  -478,  -478,  -478,  -478,   758,   759,
    -478,   760,   761,  -478,  -478,  -478,   763,  -478,   764,  -478,
     793,  -478,   810,  -478,   818,  -478,   820,  -478,   821,  -478,
     829,  -478,   830,  -478,   831,  -478,   832,  -478,   833,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,   834,  -478,   835,   841,  -478,   845,   849,  -478,   850,
     878,  -478,   882,   896,  -478,  -478,  -478,  -478,   899,  -478,
     900,  -478,   902,  -478,   903,  -478,   904,  -478,   906,  -478,
    -478,  -478,   907,  -478,   935,  -478,   939,  -478,   953,  -478,
     956,  -478,   957,  -478,   959,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,   960,  -478,   961,  -478,   963,  -478,   964,
    -478,   992,  -478,   996,  -478,  1010,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  1013,
    -478,  1014,  -478,  -478,  -478,  1016,  -478,  -478,   846,  -478,
    -478,   613,  -478,   451,  -478,   863,   920,  -478,  -478,  -478,
     977,  -478,  -478,  -478,   979,  -478,  -478,    11,    12,     5,
      26,   741,   884,   978,   941,   998,  1021,   984,  1012,   451,
    1782,  1782,  1782,  1782,  1782,  1782,  1782,  1782,  1782,  1782,
    1782,  1812,    27,  1839,  1839,  1839,  1839,  1839,  1839,  1839,
    1839,  1839,  1839,  1782,  1782,  1055,  1027,  1812,  1782,  1782,
    1782,  1782,  1782,  1782,  1782,  1782,  1782,  1782,   470,   470,
      28,   641,   470,   470,  1782,  1782,  1782,    -9,    42,    43,
      68,   128,   129,  1812,  1782,  1782,  1812,  1812,  1812,   525,
     771,  1849,  1859,  1812,  1812,  1812,  1782,  1782,  1782,  1782,
    1782,  1782,  1782,  1782,  1782,  1782,  -478,  -478,  1032,  -478,
    -478,  -478,  -478,    15,    15,    15,    15,  -478,  -478,  -478,
    -478,  -478,  -478,  1053,  -478,  -478,  -478,  -478,  -478,  1033,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  1073,  1074,  1075,  1077,
    1078,  1106,  -478,  -478,  -478,  -478,  1110,  1124,  1127,  1128,
    1130,  1131,  -478,  -478,  1132,  1134,  1135,  1163,  1167,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  1181,  1184,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  1112,  -478,   470,   470,
    1782,  1782,  1782,  1782,   470,   470,  1782,  1782,   470,   470,
    1782,  1782,  1089,  1145,  1824,  1146,  1146,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  1187,  1188,  1188,  1169,  1149,
    -478,  1191,  1193,  -478
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -478,  -478,  -478,    61,   -82,   -27,  -478,  -254,  -252,  -100,
    -105,  -477,  -478,  -478,  -478,  -478,  -208,   929,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,   429,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,  -478,
    -478,  -478,  -478,  -478
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -357
static const yytype_int16 yytable[] =
{
     283,   285,   401,   403,   545,   544,   544,   530,   272,   657,
     552,   255,   544,   303,   305,   653,   655,   383,   256,   770,
     384,   330,   332,   334,   336,   338,   340,   342,   344,   346,
     659,   682,   721,   730,   523,   257,   731,   355,   531,   362,
     364,   366,   368,   370,   372,   374,   376,   378,   380,   382,
     658,   546,   547,   548,   395,   550,   392,   273,   554,   517,
     518,   385,   556,   258,   654,   656,   557,   558,   771,   428,
     276,   660,   683,   259,   277,   278,   279,   280,   281,   282,
     451,   356,   306,   260,   732,   734,   468,   733,   735,   474,
     476,   478,   480,   482,   484,   486,   488,   302,   562,   563,
     393,   497,   499,   501,   320,   322,   324,   326,   328,   261,
     736,   519,   522,   737,   386,   529,   516,   387,   348,   350,
     351,   353,  -346,   276,   358,   360,   307,   277,   278,   279,
     280,   281,   282,   308,   309,   310,   311,   312,   276,   455,
     313,   262,   277,   278,   279,   280,   281,   282,   263,   537,
     414,   416,   418,   420,   422,   424,   426,   276,   388,   264,
     265,   277,   278,   279,   280,   281,   282,   449,   517,   518,
     738,   740,   266,   739,   741,   470,   472,   773,   774,   775,
     456,   267,   354,   457,  -348,   490,   492,   494,   439,   495,
    -352,  -356,   503,   505,   507,   509,   511,   513,   515,   268,
     269,   525,   527,    40,    41,   533,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,   140,   141,   142,   143,   144,   145,
     270,   271,   274,   275,   291,   292,   396,   293,     2,     3,
     389,     4,   304,   390,   294,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,   648,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
     397,   669,    39,   276,   391,   295,   297,   277,   278,   279,
     280,   281,   282,   299,   300,    40,    41,   301,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   143,
     144,   145,   400,   685,   687,   689,   691,   693,   695,   697,
     699,   701,   703,   536,   404,   407,   540,   405,   408,   681,
     539,   549,   553,   559,   410,   440,   443,   411,   441,   444,
     452,   398,   276,   453,   555,   708,   277,   278,   279,   280,
     281,   282,   560,   276,   429,   431,   307,   277,   278,   279,
     280,   281,   282,   308,   309,   310,   311,   312,   406,   409,
     814,   742,   544,   433,   745,   746,   747,   521,   412,   442,
     445,   756,   757,   758,   454,   399,   684,   686,   688,   690,
     692,   694,   696,   698,   700,   702,   319,   276,   430,   432,
     748,   277,   278,   279,   280,   281,   282,   561,   435,   437,
     564,   719,   720,   722,   724,   725,   726,   434,   276,   565,
     566,   567,   277,   278,   279,   280,   281,   282,   568,   517,
     518,   329,   749,   751,   753,   755,   569,   276,   570,   446,
     307,   277,   278,   279,   280,   281,   282,   308,   309,   310,
     311,   312,   436,   438,   313,   571,   572,   573,   534,   458,
     461,   670,   671,   672,   673,   674,   675,   676,   677,   678,
     679,   680,   276,   574,   575,   723,   277,   278,   279,   280,
     281,   282,   464,   447,   704,   705,   647,   576,   577,   709,
     710,   711,   712,   713,   714,   715,   716,   717,   718,   321,
     459,   462,   535,   460,   463,   727,   728,   729,   578,   579,
     580,   581,   582,   276,   323,   743,   744,   277,   278,   279,
     280,   281,   282,   465,   583,   584,   466,   759,   760,   761,
     762,   763,   764,   765,   766,   767,   768,   585,   586,   587,
     276,   588,   589,   307,   277,   278,   279,   280,   281,   282,
     308,   309,   310,   311,   312,   276,   325,   313,   307,   277,
     278,   279,   280,   281,   282,   308,   309,   310,   311,   312,
     590,   327,   313,   286,   591,   661,   287,   592,   593,   594,
     595,   798,   799,   596,   288,   289,   347,   804,   805,   597,
     290,   808,   809,   598,   599,   600,   601,   276,   602,   603,
     307,   277,   278,   279,   280,   281,   282,   308,   309,   310,
     311,   312,   276,   331,   313,   307,   277,   278,   279,   280,
     281,   282,   308,   309,   310,   311,   312,   276,   604,   313,
     307,   277,   278,   279,   280,   281,   282,   308,   309,   310,
     311,   312,   349,   276,   313,   605,   750,   277,   278,   279,
     280,   281,   282,   606,   276,   607,   608,   352,   277,   278,
     279,   280,   281,   282,   609,   610,   611,   612,   613,   614,
     615,   800,   801,   802,   803,   357,   616,   806,   807,   646,
     617,   810,   811,   276,   618,   619,   307,   277,   278,   279,
     280,   281,   282,   308,   309,   310,   311,   312,   276,   359,
     313,   307,   277,   278,   279,   280,   281,   282,   308,   309,
     310,   311,   312,   620,   413,   313,   276,   621,   662,   307,
     277,   278,   279,   280,   281,   282,   308,   309,   310,   311,
     312,   622,   415,   313,   623,   624,   649,   625,   626,   627,
     276,   628,   629,   307,   277,   278,   279,   280,   281,   282,
     308,   309,   310,   311,   312,   276,   417,   313,   307,   277,
     278,   279,   280,   281,   282,   308,   309,   310,   311,   312,
     630,   419,   313,   276,   631,   664,   307,   277,   278,   279,
     280,   281,   282,   308,   309,   310,   311,   312,   632,   421,
     313,   633,   634,   650,   635,   636,   637,   276,   638,   639,
     307,   277,   278,   279,   280,   281,   282,   308,   309,   310,
     311,   312,   276,   423,   313,   307,   277,   278,   279,   280,
     281,   282,   308,   309,   310,   311,   312,   640,   425,   313,
     276,   641,   665,   307,   277,   278,   279,   280,   281,   282,
     308,   309,   310,   311,   312,   642,   448,   313,   643,   644,
     651,   645,   652,   663,   276,   666,   667,   307,   277,   278,
     279,   280,   281,   282,   308,   309,   310,   311,   312,   276,
     469,   313,   307,   277,   278,   279,   280,   281,   282,   308,
     309,   310,   311,   312,   668,   471,   313,   276,   776,   706,
     307,   277,   278,   279,   280,   281,   282,   308,   309,   310,
     311,   312,   707,   489,   313,   769,   777,   538,   778,   779,
     780,   276,   781,   782,   307,   277,   278,   279,   280,   281,
     282,   308,   309,   310,   311,   312,   276,   491,   313,   307,
     277,   278,   279,   280,   281,   282,   308,   309,   310,   311,
     312,   783,   493,   313,   276,   784,   797,   307,   277,   278,
     279,   280,   281,   282,   308,   309,   310,   311,   312,   785,
     502,   313,   786,   787,   812,   788,   789,   790,   276,   791,
     792,   307,   277,   278,   279,   280,   281,   282,   308,   309,
     310,   311,   312,   276,   504,   313,   307,   277,   278,   279,
     280,   281,   282,   308,   309,   310,   311,   312,   793,   506,
     313,   276,   794,   820,   307,   277,   278,   279,   280,   281,
     282,   308,   309,   310,   311,   312,   795,   508,   313,   796,
     813,   815,   818,   819,   821,   276,   822,   823,   307,   277,
     278,   279,   280,   281,   282,   308,   309,   310,   311,   312,
     276,   510,   313,   307,   277,   278,   279,   280,   281,   282,
     308,   309,   310,   311,   312,   817,   512,   313,   276,     0,
       0,   307,   277,   278,   279,   280,   281,   282,   308,   309,
     310,   311,   312,     0,   514,   313,     0,     0,     0,     0,
       0,     0,   276,     0,     0,   307,   277,   278,   279,   280,
     281,   282,   308,   309,   310,   311,   312,   276,   524,   313,
     307,   277,   278,   279,   280,   281,   282,   308,   309,   310,
     311,   312,     0,   526,   313,   276,     0,     0,   307,   277,
     278,   279,   280,   281,   282,   308,   309,   310,   311,   312,
       0,   532,   313,     0,     0,     0,     0,     0,     0,   276,
       0,     0,   307,   277,   278,   279,   280,   281,   282,   308,
     309,   310,   311,   312,   276,   402,   313,   307,   277,   278,
     279,   280,   281,   282,   308,   309,   310,   311,   312,     0,
     333,   313,   276,     0,     0,   307,   277,   278,   279,   280,
     281,   282,   308,   309,   310,   311,   312,   335,     0,   313,
       0,     0,     0,     0,     0,     0,   276,   337,     0,   307,
     277,   278,   279,   280,   281,   282,   308,   309,   310,   311,
     312,   276,   339,     0,     0,   277,   278,   279,   280,   281,
     282,     0,   341,     0,     0,     0,     0,     0,   276,     0,
       0,     0,   277,   278,   279,   280,   281,   282,   276,   343,
       0,     0,   277,   278,   279,   280,   281,   282,     0,   345,
       0,     0,     0,   276,     0,     0,     0,   277,   278,   279,
     280,   281,   282,   276,   361,     0,     0,   277,   278,   279,
     280,   281,   282,     0,   363,     0,     0,     0,     0,     0,
     276,     0,     0,     0,   277,   278,   279,   280,   281,   282,
     276,   365,     0,     0,   277,   278,   279,   280,   281,   282,
       0,   367,     0,     0,     0,   276,     0,     0,     0,   277,
     278,   279,   280,   281,   282,   276,   369,     0,     0,   277,
     278,   279,   280,   281,   282,     0,   371,     0,     0,     0,
       0,     0,   276,     0,     0,     0,   277,   278,   279,   280,
     281,   282,   276,   373,     0,     0,   277,   278,   279,   280,
     281,   282,     0,   375,     0,     0,     0,   276,     0,     0,
       0,   277,   278,   279,   280,   281,   282,   276,   377,     0,
       0,   277,   278,   279,   280,   281,   282,     0,   379,     0,
       0,     0,     0,     0,   276,     0,     0,     0,   277,   278,
     279,   280,   281,   282,   276,   381,     0,     0,   277,   278,
     279,   280,   281,   282,     0,   394,     0,     0,     0,   276,
       0,     0,     0,   277,   278,   279,   280,   281,   282,   276,
     427,     0,     0,   277,   278,   279,   280,   281,   282,     0,
     450,     0,     0,     0,     0,     0,   276,     0,     0,     0,
     277,   278,   279,   280,   281,   282,   276,   467,     0,     0,
     277,   278,   279,   280,   281,   282,     0,   473,     0,     0,
       0,   276,     0,     0,     0,   277,   278,   279,   280,   281,
     282,   276,   475,     0,     0,   277,   278,   279,   280,   281,
     282,     0,   477,     0,     0,     0,     0,     0,   276,     0,
       0,     0,   277,   278,   279,   280,   281,   282,   276,   479,
       0,     0,   277,   278,   279,   280,   281,   282,     0,   481,
       0,     0,     0,   276,     0,     0,     0,   277,   278,   279,
     280,   281,   282,   276,   483,     0,     0,   277,   278,   279,
     280,   281,   282,     0,   485,     0,     0,     0,     0,     0,
     276,     0,     0,     0,   277,   278,   279,   280,   281,   282,
     276,   487,     0,     0,   277,   278,   279,   280,   281,   282,
       0,   496,     0,     0,     0,   276,     0,     0,     0,   277,
     278,   279,   280,   281,   282,   276,   498,     0,     0,   277,
     278,   279,   280,   281,   282,     0,   500,     0,     0,     0,
       0,     0,   276,     0,     0,     0,   277,   278,   279,   280,
     281,   282,   276,   528,     0,     0,   277,   278,   279,   280,
     281,   282,     0,     0,     0,     0,     0,   276,     0,     0,
       0,   277,   278,   279,   280,   281,   282,   276,     0,     0,
       0,   277,   278,   279,   280,   281,   282,     0,     0,     0,
       0,     0,     0,     0,   276,     0,     0,     0,   277,   278,
     279,   280,   281,   282,   276,     0,     0,   307,   277,   278,
     279,   280,   281,   282,   308,   309,   310,   311,   312,   551,
       0,   313,   307,     0,     0,     0,     0,     0,     0,   308,
     309,   310,   311,   312,   276,     0,   542,   307,   277,   278,
     279,   280,   281,   282,   308,   309,   310,   311,   312,   307,
       0,     0,     0,     0,     0,     0,   308,   309,   310,   311,
     312,   276,     0,   542,   307,   277,   278,   279,   280,   281,
     282,   276,   309,   310,   752,   277,   278,   279,   280,   281,
     282,   276,     0,     0,   754,   277,   278,   279,   280,   281,
     282
};

static const yytype_int16 yycheck[] =
{
      27,    28,    84,    85,   258,   257,   258,     1,     4,     4,
     264,     3,   264,    40,    41,     4,     4,     1,     4,     4,
       4,    48,    49,    50,    51,    52,    53,    54,    55,    56,
       4,     4,     4,    42,   139,     4,    45,     1,   143,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      45,   259,   260,   261,    81,   263,     1,    53,   266,    53,
      54,    45,   270,     4,    53,    53,   274,   275,    53,    96,
      42,    45,    45,     4,    46,    47,    48,    49,    50,    51,
     107,    45,     1,     4,    42,    42,   113,    45,    45,   116,
     117,   118,   119,   120,   121,   122,   123,     1,     5,     6,
      45,   128,   129,   130,    43,    44,    45,    46,    47,     4,
      42,   138,   139,    45,     1,   142,     1,     4,    57,    58,
       1,    60,     3,    42,    63,    64,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    42,     1,
      59,     4,    46,    47,    48,    49,    50,    51,     4,     3,
      89,    90,    91,    92,    93,    94,    95,    42,    45,     4,
       4,    46,    47,    48,    49,    50,    51,   106,    53,    54,
      42,    42,     4,    45,    45,   114,   115,   654,   655,   656,
      42,     4,     1,    45,     3,   124,   125,   126,     1,     1,
       3,     3,   131,   132,   133,   134,   135,   136,   137,    42,
       4,   140,   141,    57,    58,   144,    60,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   147,   148,   149,   150,   151,   152,   153,
     154,   155,   156,   157,   158,   159,   160,   161,   162,   163,
       4,     4,     4,     4,    45,    53,     1,    53,     0,     1,
       1,     3,     1,     4,    53,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,   543,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      45,   569,    44,    42,    45,    42,    42,    46,    47,    48,
      49,    50,    51,    42,    42,    57,    58,     6,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,     1,   583,   584,   585,   586,   587,   588,   589,
     590,   591,   592,     3,     1,     1,    45,     4,     4,   581,
       3,    43,    42,     5,     1,     1,     1,     4,     4,     4,
       1,     1,    42,     4,    43,   597,    46,    47,    48,    49,
      50,    51,     5,    42,     1,     1,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    45,    45,
     794,   623,   794,     1,   626,   627,   628,     1,    45,    45,
      45,   633,   634,   635,    45,    45,   583,   584,   585,   586,
     587,   588,   589,   590,   591,   592,     1,    42,    45,    45,
      45,    46,    47,    48,    49,    50,    51,     5,     1,     1,
       5,   608,   609,   610,   611,   612,   613,    45,    42,     5,
       5,     5,    46,    47,    48,    49,    50,    51,     5,    53,
      54,     1,   629,   630,   631,   632,     4,    42,     5,     1,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    45,    45,    59,     5,     5,     5,     1,     1,
       1,   570,   571,   572,   573,   574,   575,   576,   577,   578,
     579,   580,    42,     5,     5,     4,    46,    47,    48,    49,
      50,    51,     1,    45,   593,   594,    43,     5,     5,   598,
     599,   600,   601,   602,   603,   604,   605,   606,   607,     1,
      42,    42,    45,    45,    45,   614,   615,   616,     5,     5,
       5,     5,     5,    42,     1,   624,   625,    46,    47,    48,
      49,    50,    51,    42,     5,     5,    45,   636,   637,   638,
     639,   640,   641,   642,   643,   644,   645,     5,     5,     5,
      42,     5,     5,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    42,     1,    59,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
       5,     1,    59,    42,     5,     4,    45,     5,     5,     5,
       5,   778,   779,     5,    53,    54,     1,   784,   785,     5,
      59,   788,   789,     5,     5,     5,     5,    42,     5,     5,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    42,     1,    59,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    42,     5,    59,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,     1,    42,    59,     5,    45,    46,    47,    48,
      49,    50,    51,     5,    42,     5,     5,     1,    46,    47,
      48,    49,    50,    51,     5,     5,     5,     5,     5,     5,
       5,   780,   781,   782,   783,     1,     5,   786,   787,     3,
       5,   790,   791,    42,     5,     5,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    42,     1,
      59,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,     5,     1,    59,    42,     5,     4,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,     5,     1,    59,     5,     5,    43,     5,     5,     5,
      42,     5,     5,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    42,     1,    59,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
       5,     1,    59,    42,     5,     4,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,     5,     1,
      59,     5,     5,    43,     5,     5,     5,    42,     5,     5,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    42,     1,    59,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,     5,     1,    59,
      42,     5,     4,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,     5,     1,    59,     5,     5,
      43,     5,    43,    45,    42,     4,    42,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    42,
       1,    59,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    42,     1,    59,    42,     5,     4,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    45,     1,    59,    43,    43,   148,     5,     5,
       5,    42,     5,     5,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    42,     1,    59,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,     5,     1,    59,    42,     5,     4,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,     5,
       1,    59,     5,     5,    45,     5,     5,     5,    42,     5,
       5,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    42,     1,    59,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,     5,     1,
      59,    42,     5,     4,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,     5,     1,    59,     5,
      45,    45,     5,     5,    45,    42,     5,     4,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      42,     1,    59,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,   796,     1,    59,    42,    -1,
      -1,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    -1,     1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    42,    -1,    -1,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    42,     1,    59,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    -1,     1,    59,    42,    -1,    -1,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      -1,     1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    42,
      -1,    -1,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    42,     1,    59,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    -1,
       1,    59,    42,    -1,    -1,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,     1,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    42,     1,    -1,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    42,     1,    -1,    -1,    46,    47,    48,    49,    50,
      51,    -1,     1,    -1,    -1,    -1,    -1,    -1,    42,    -1,
      -1,    -1,    46,    47,    48,    49,    50,    51,    42,     1,
      -1,    -1,    46,    47,    48,    49,    50,    51,    -1,     1,
      -1,    -1,    -1,    42,    -1,    -1,    -1,    46,    47,    48,
      49,    50,    51,    42,     1,    -1,    -1,    46,    47,    48,
      49,    50,    51,    -1,     1,    -1,    -1,    -1,    -1,    -1,
      42,    -1,    -1,    -1,    46,    47,    48,    49,    50,    51,
      42,     1,    -1,    -1,    46,    47,    48,    49,    50,    51,
      -1,     1,    -1,    -1,    -1,    42,    -1,    -1,    -1,    46,
      47,    48,    49,    50,    51,    42,     1,    -1,    -1,    46,
      47,    48,    49,    50,    51,    -1,     1,    -1,    -1,    -1,
      -1,    -1,    42,    -1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    42,     1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    -1,     1,    -1,    -1,    -1,    42,    -1,    -1,
      -1,    46,    47,    48,    49,    50,    51,    42,     1,    -1,
      -1,    46,    47,    48,    49,    50,    51,    -1,     1,    -1,
      -1,    -1,    -1,    -1,    42,    -1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    42,     1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    -1,     1,    -1,    -1,    -1,    42,
      -1,    -1,    -1,    46,    47,    48,    49,    50,    51,    42,
       1,    -1,    -1,    46,    47,    48,    49,    50,    51,    -1,
       1,    -1,    -1,    -1,    -1,    -1,    42,    -1,    -1,    -1,
      46,    47,    48,    49,    50,    51,    42,     1,    -1,    -1,
      46,    47,    48,    49,    50,    51,    -1,     1,    -1,    -1,
      -1,    42,    -1,    -1,    -1,    46,    47,    48,    49,    50,
      51,    42,     1,    -1,    -1,    46,    47,    48,    49,    50,
      51,    -1,     1,    -1,    -1,    -1,    -1,    -1,    42,    -1,
      -1,    -1,    46,    47,    48,    49,    50,    51,    42,     1,
      -1,    -1,    46,    47,    48,    49,    50,    51,    -1,     1,
      -1,    -1,    -1,    42,    -1,    -1,    -1,    46,    47,    48,
      49,    50,    51,    42,     1,    -1,    -1,    46,    47,    48,
      49,    50,    51,    -1,     1,    -1,    -1,    -1,    -1,    -1,
      42,    -1,    -1,    -1,    46,    47,    48,    49,    50,    51,
      42,     1,    -1,    -1,    46,    47,    48,    49,    50,    51,
      -1,     1,    -1,    -1,    -1,    42,    -1,    -1,    -1,    46,
      47,    48,    49,    50,    51,    42,     1,    -1,    -1,    46,
      47,    48,    49,    50,    51,    -1,     1,    -1,    -1,    -1,
      -1,    -1,    42,    -1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    42,     1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    -1,    -1,    -1,    -1,    -1,    42,    -1,    -1,
      -1,    46,    47,    48,    49,    50,    51,    42,    -1,    -1,
      -1,    46,    47,    48,    49,    50,    51,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    42,    -1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    42,    -1,    -1,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    42,
      -1,    59,    45,    -1,    -1,    -1,    -1,    -1,    -1,    52,
      53,    54,    55,    56,    42,    -1,    59,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    45,
      -1,    -1,    -1,    -1,    -1,    -1,    52,    53,    54,    55,
      56,    42,    -1,    59,    45,    46,    47,    48,    49,    50,
      51,    42,    53,    54,    45,    46,    47,    48,    49,    50,
      51,    42,    -1,    -1,    45,    46,    47,    48,    49,    50,
      51
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   165,     0,     1,     3,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    44,
      57,    58,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
     148,   149,   150,   151,   152,   153,   154,   155,   156,   157,
     158,   159,   160,   161,   162,   163,   166,   176,   179,   181,
     182,   183,   184,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,     3,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,    42,     4,
       4,     4,     4,    53,     4,     4,    42,    46,    47,    48,
      49,    50,    51,   169,   170,   169,    42,    45,    53,    54,
      59,    45,    53,    53,    53,    42,   177,    42,   178,    42,
      42,     6,     1,   169,     1,   169,     1,    45,    52,    53,
      54,    55,    56,    59,   167,   168,   169,   172,   173,     1,
     167,     1,   167,     1,   167,     1,   167,     1,   167,     1,
     169,     1,   169,     1,   169,     1,   169,     1,   169,     1,
     169,     1,   169,     1,   169,     1,   169,     1,   167,     1,
     167,     1,     1,   167,     1,     1,    45,     1,   167,     1,
     167,     1,   169,     1,   169,     1,   169,     1,   169,     1,
     169,     1,   169,     1,   169,     1,   169,     1,   169,     1,
     169,     1,   169,     1,     4,    45,     1,     4,    45,     1,
       4,    45,     1,    45,     1,   169,     1,    45,     1,    45,
       1,   168,     1,   168,     1,     4,    45,     1,     4,    45,
       1,     4,    45,     1,   167,     1,   167,     1,   167,     1,
     167,     1,   167,     1,   167,     1,   167,     1,   169,     1,
      45,     1,    45,     1,    45,     1,    45,     1,    45,     1,
       1,     4,    45,     1,     4,    45,     1,    45,     1,   167,
       1,   169,     1,     4,    45,     1,    42,    45,     1,    42,
      45,     1,    42,    45,     1,    42,    45,     1,   169,     1,
     167,     1,   167,     1,   169,     1,   169,     1,   169,     1,
     169,     1,   169,     1,   169,     1,   169,     1,   169,     1,
     167,     1,   167,     1,   167,     1,     1,   169,     1,   169,
       1,   169,     1,   167,     1,   167,     1,   167,     1,   167,
       1,   167,     1,   167,     1,   167,     1,    53,    54,   169,
     174,     1,   169,   174,     1,   167,     1,   167,     1,   169,
       1,   174,     1,   167,     1,    45,     3,     3,   181,     3,
      45,   180,    59,   171,   172,   171,   180,   180,   180,    43,
     180,    42,   171,    42,   180,    43,   180,   180,   180,     5,
       5,     5,     5,     6,     5,     5,     5,     5,     5,     4,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     3,    43,   180,    43,
      43,    43,    43,     4,    53,     4,    53,     4,    45,     4,
      45,     4,     4,    45,     4,     4,     4,    42,    42,   180,
     167,   167,   167,   167,   167,   167,   167,   167,   167,   167,
     167,   168,     4,    45,   169,   173,   169,   173,   169,   173,
     169,   173,   169,   173,   169,   173,   169,   173,   169,   173,
     169,   173,   169,   173,   167,   167,     4,    45,   168,   167,
     167,   167,   167,   167,   167,   167,   167,   167,   167,   169,
     169,     4,   169,     4,   169,   169,   169,   167,   167,   167,
      42,    45,    42,    45,    42,    45,    42,    45,    42,    45,
      42,    45,   168,   167,   167,   168,   168,   168,    45,   169,
      45,   169,    45,   169,    45,   169,   168,   168,   168,   167,
     167,   167,   167,   167,   167,   167,   167,   167,   167,    43,
       4,    53,   175,   175,   175,   175,     5,    43,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     4,   169,   169,
     167,   167,   167,   167,   169,   169,   167,   167,   169,   169,
     167,   167,    45,    45,   171,    45,   250,   250,     5,     5,
       4,    45,     5,     4
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */






/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  /* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;

  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 9:
#line 245 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax, LINE - 1 ); }
    break;

  case 35:
#line 261 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addEntry(); }
    break;

  case 36:
#line 262 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->setModuleName( (yyvsp[(2) - (2)]) ); }
    break;

  case 37:
#line 263 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 38:
#line 264 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 39:
#line 265 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 40:
#line 266 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 41:
#line 267 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 42:
#line 268 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 43:
#line 269 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 44:
#line 270 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 45:
#line 271 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLocal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 46:
#line 272 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addParam( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 47:
#line 273 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 48:
#line 274 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 49:
#line 275 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 50:
#line 276 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 51:
#line 277 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); }
    break;

  case 52:
#line 278 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]), false ); }
    break;

  case 53:
#line 279 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]), true ); }
    break;

  case 54:
#line 280 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 55:
#line 281 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]), 0, false ); }
    break;

  case 56:
#line 282 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]), 0, true ); }
    break;

  case 57:
#line 283 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addImport( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), false );
   }
    break;

  case 58:
#line 286 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addImport( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), true );
   }
    break;

  case 59:
#line 289 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addAlias( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), true );
   }
    break;

  case 60:
#line 292 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addAlias( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), false );
   }
    break;

  case 61:
#line 295 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 62:
#line 296 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 63:
#line 297 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 64:
#line 298 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 65:
#line 299 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 66:
#line 300 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 67:
#line 301 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 68:
#line 302 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 69:
#line 303 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 70:
#line 304 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (6)]), (yyvsp[(6) - (6)]), (yyvsp[(4) - (6)]) ); }
    break;

  case 71:
#line 305 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDEndSwitch(); }
    break;

  case 72:
#line 306 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 73:
#line 307 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 74:
#line 308 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addPropRef( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 77:
#line 311 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 78:
#line 312 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 79:
#line 313 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 80:
#line 314 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 81:
#line 315 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 82:
#line 316 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 83:
#line 317 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassCtor( (yyvsp[(2) - (2)]) ); }
    break;

  case 84:
#line 318 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); /* Currently the same as .endfunc */ }
    break;

  case 85:
#line 319 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInherit((yyvsp[(2) - (2)])); }
    break;

  case 86:
#line 320 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFrom( (yyvsp[(2) - (2)]) ); }
    break;

  case 87:
#line 321 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addExtern( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 88:
#line 322 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addDLine( (yyvsp[(2) - (2)]) ); }
    break;

  case 89:
#line 324 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 90:
#line 329 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         (yyvsp[(2) - (2)])->asString().exported( true );
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 91:
#line 335 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 92:
#line 343 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(1) - (1)]) ); }
    break;

  case 93:
#line 344 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(3) - (3)]) ); }
    break;

  case 94:
#line 348 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(1) - (1)]) ); }
    break;

  case 95:
#line 349 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(3) - (3)]) ); }
    break;

  case 96:
#line 352 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->defineLabel( (yyvsp[(1) - (2)])->asLabel() ); }
    break;

  case 97:
#line 356 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {(yyval) = new Falcon::Pseudo( LINE, (Falcon::int64) 0 ); }
    break;

  case 204:
#line 469 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 205:
#line 470 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LD" ); }
    break;

  case 206:
#line 474 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDRF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 207:
#line 475 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDRF" ); }
    break;

  case 208:
#line 479 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LNIL, (yyvsp[(2) - (2)]) ); }
    break;

  case 209:
#line 480 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LNIL" ); }
    break;

  case 210:
#line 484 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 211:
#line 485 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADD" ); }
    break;

  case 212:
#line 489 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 213:
#line 490 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADDS" ); }
    break;

  case 214:
#line 495 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 215:
#line 496 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUB" ); }
    break;

  case 216:
#line 500 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUBS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 217:
#line 501 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUBS" ); }
    break;

  case 218:
#line 505 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MUL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 219:
#line 506 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MUL" ); }
    break;

  case 220:
#line 510 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MULS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 221:
#line 511 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MULS" ); }
    break;

  case 222:
#line 516 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 223:
#line 517 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIV" ); }
    break;

  case 224:
#line 521 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 225:
#line 522 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIVS" ); }
    break;

  case 226:
#line 526 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MOD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 227:
#line 527 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MOD" ); }
    break;

  case 228:
#line 531 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POW, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 229:
#line 532 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POW" ); }
    break;

  case 230:
#line 537 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 231:
#line 538 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EQ" ); }
    break;

  case 232:
#line 542 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 233:
#line 543 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEQ" ); }
    break;

  case 234:
#line 547 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 235:
#line 548 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GE" ); }
    break;

  case 236:
#line 552 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 237:
#line 553 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GT" ); }
    break;

  case 238:
#line 557 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 239:
#line 558 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LE" ); }
    break;

  case 240:
#line 562 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 241:
#line 563 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LT" ); }
    break;

  case 242:
#line 567 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 243:
#line 568 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 244:
#line 569 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRY" ); }
    break;

  case 245:
#line 573 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INC, (yyvsp[(2) - (2)]) ); }
    break;

  case 246:
#line 574 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INC" ); }
    break;

  case 247:
#line 578 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DEC, (yyvsp[(2) - (2)])  ); }
    break;

  case 248:
#line 579 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DEC" ); }
    break;

  case 249:
#line 584 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INCP, (yyvsp[(2) - (2)]) ); }
    break;

  case 250:
#line 585 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INCP" ); }
    break;

  case 251:
#line 589 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DECP, (yyvsp[(2) - (2)])  ); }
    break;

  case 252:
#line 590 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DECP" ); }
    break;

  case 253:
#line 595 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEG, (yyvsp[(2) - (2)])  ); }
    break;

  case 254:
#line 596 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEG" ); }
    break;

  case 255:
#line 600 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOT, (yyvsp[(2) - (2)])  ); }
    break;

  case 256:
#line 601 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOT" ); }
    break;

  case 257:
#line 605 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 258:
#line 606 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 259:
#line 607 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "CALL" ); }
    break;

  case 260:
#line 611 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 261:
#line 612 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 262:
#line 613 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INST" ); }
    break;

  case 263:
#line 617 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_UNPK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 264:
#line 618 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPK" ); }
    break;

  case 265:
#line 622 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_UNPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 266:
#line 623 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPS" ); }
    break;

  case 267:
#line 628 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PUSH, (yyvsp[(2) - (2)]) ); }
    break;

  case 268:
#line 629 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHN ); }
    break;

  case 269:
#line 630 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PUSH" ); }
    break;

  case 270:
#line 634 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHR, (yyvsp[(2) - (2)]) ); }
    break;

  case 271:
#line 635 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSHR" ); }
    break;

  case 272:
#line 640 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_POP, (yyvsp[(2) - (2)]) ); }
    break;

  case 273:
#line 641 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POP" ); }
    break;

  case 274:
#line 645 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PEEK, (yyvsp[(2) - (2)]) ); }
    break;

  case 275:
#line 646 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PEEK" ); }
    break;

  case 276:
#line 650 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 277:
#line 651 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XPOP" ); }
    break;

  case 278:
#line 656 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 279:
#line 657 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 280:
#line 658 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDV" ); }
    break;

  case 281:
#line 662 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 282:
#line 663 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 283:
#line 664 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVT" ); }
    break;

  case 284:
#line 668 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 285:
#line 669 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 286:
#line 670 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STV" ); }
    break;

  case 287:
#line 674 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 288:
#line 675 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 289:
#line 676 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVR" ); }
    break;

  case 290:
#line 680 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 291:
#line 681 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 292:
#line 682 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVS" ); }
    break;

  case 293:
#line 686 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 294:
#line 687 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 295:
#line 688 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDP" ); yyerrok; }
    break;

  case 296:
#line 692 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 297:
#line 693 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 298:
#line 694 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPT" ); yyerrok; }
    break;

  case 299:
#line 698 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 300:
#line 699 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 301:
#line 700 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STP" ); }
    break;

  case 302:
#line 704 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 303:
#line 705 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 304:
#line 706 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPR" ); }
    break;

  case 305:
#line 710 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 306:
#line 711 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 307:
#line 712 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPS" ); }
    break;

  case 308:
#line 716 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 309:
#line 717 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 310:
#line 718 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAV" ); }
    break;

  case 311:
#line 722 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 312:
#line 723 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 313:
#line 724 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAN" ); }
    break;

  case 314:
#line 728 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 315:
#line 729 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 316:
#line 730 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAL" ); }
    break;

  case 317:
#line 734 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_IPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 318:
#line 735 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IPOP" ); }
    break;

  case 319:
#line 739 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GENA, (yyvsp[(2) - (2)]) ); }
    break;

  case 320:
#line 740 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENA" ); }
    break;

  case 321:
#line 744 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GEND, (yyvsp[(2) - (2)]) ); }
    break;

  case 322:
#line 745 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEND" ); }
    break;

  case 323:
#line 749 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GENR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 324:
#line 750 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENR" ); }
    break;

  case 325:
#line 754 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GEOR, (yyvsp[(2) - (2)]) ); }
    break;

  case 326:
#line 755 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEOR" ); }
    break;

  case 327:
#line 759 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RIS, (yyvsp[(2) - (2)]) ); }
    break;

  case 328:
#line 760 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RIS" ); }
    break;

  case 329:
#line 764 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 330:
#line 765 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 331:
#line 766 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JMP" ); }
    break;

  case 332:
#line 770 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOOL, (yyvsp[(1) - (2)]) ); }
    break;

  case 333:
#line 771 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOOL" ); }
    break;

  case 334:
#line 775 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 335:
#line 776 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 336:
#line 777 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFT" ); }
    break;

  case 337:
#line 781 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 338:
#line 782 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 339:
#line 783 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFF" ); }
    break;

  case 340:
#line 788 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 341:
#line 789 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 342:
#line 790 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORK" ); }
    break;

  case 343:
#line 794 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 344:
#line 795 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 345:
#line 796 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JTRY" ); }
    break;

  case 346:
#line 800 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RET ); }
    break;

  case 347:
#line 801 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RET" ); }
    break;

  case 348:
#line 805 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETA ); }
    break;

  case 349:
#line 806 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETA" ); }
    break;

  case 350:
#line 810 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETV, (yyvsp[(2) - (2)]) ); }
    break;

  case 351:
#line 811 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETV" ); }
    break;

  case 352:
#line 815 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOP ); }
    break;

  case 353:
#line 816 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOP" ); }
    break;

  case 354:
#line 820 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_PTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 355:
#line 821 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PTRY" ); }
    break;

  case 356:
#line 825 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_END ); }
    break;

  case 357:
#line 826 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "END" ); }
    break;

  case 358:
#line 830 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 359:
#line 831 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SWCH" ); }
    break;

  case 360:
#line 835 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 361:
#line 836 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SELE" ); }
    break;

  case 362:
#line 841 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         Falcon::Pseudo *psd = new Falcon::Pseudo( Falcon::Pseudo::tswitch_list );
         psd->line( LINE );
         psd->asList()->pushBack( (yyvsp[(1) - (3)]) );
         psd->asList()->pushBack( (yyvsp[(3) - (3)]) );
         (yyval) = psd;
      }
    break;

  case 363:
#line 850 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(3) - (5)]) );
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(5) - (5)]) );
         (yyval) = (yyvsp[(1) - (5)]);
      }
    break;

  case 364:
#line 858 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 365:
#line 859 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 366:
#line 860 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ONCE" ); }
    break;

  case 367:
#line 864 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 368:
#line 865 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 369:
#line 866 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 370:
#line 867 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 371:
#line 868 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BAND" ); }
    break;

  case 372:
#line 872 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 373:
#line 873 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 374:
#line 874 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 375:
#line 875 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 376:
#line 876 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOR" ); }
    break;

  case 377:
#line 880 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 378:
#line 881 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 379:
#line 882 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 380:
#line 883 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 381:
#line 884 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 382:
#line 888 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 383:
#line 889 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 384:
#line 890 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 385:
#line 894 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_AND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 386:
#line 895 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "AND" ); }
    break;

  case 387:
#line 899 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_OR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 388:
#line 900 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "OR" ); }
    break;

  case 389:
#line 904 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ANDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 390:
#line 905 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ANDS" ); }
    break;

  case 391:
#line 909 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 392:
#line 910 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ORS" ); }
    break;

  case 393:
#line 914 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 394:
#line 915 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XORS" ); }
    break;

  case 395:
#line 919 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MODS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 396:
#line 920 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MODS" ); }
    break;

  case 397:
#line 924 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POWS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 398:
#line 925 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POWS" ); }
    break;

  case 399:
#line 929 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOTS, (yyvsp[(2) - (2)]) ); }
    break;

  case 400:
#line 930 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOTS" ); }
    break;

  case 401:
#line 934 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 402:
#line 935 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 403:
#line 936 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HAS" ); }
    break;

  case 404:
#line 940 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 405:
#line 941 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 406:
#line 942 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HASN" ); }
    break;

  case 407:
#line 946 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 408:
#line 947 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 409:
#line 948 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVE" ); }
    break;

  case 410:
#line 952 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 411:
#line 953 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 412:
#line 954 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVN" ); }
    break;

  case 413:
#line 959 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_IN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 414:
#line 960 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IN" ); }
    break;

  case 415:
#line 964 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOIN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 416:
#line 965 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOIN" ); }
    break;

  case 417:
#line 969 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PROV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 418:
#line 970 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PROV" ); }
    break;

  case 419:
#line 974 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSIN, (yyvsp[(2) - (2)]) ); }
    break;

  case 420:
#line 975 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSIN" ); }
    break;

  case 421:
#line 979 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PASS, (yyvsp[(2) - (2)]) ); }
    break;

  case 422:
#line 980 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PASS" ); }
    break;

  case 423:
#line 984 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 424:
#line 985 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHR" ); }
    break;

  case 425:
#line 989 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 426:
#line 990 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHL" ); }
    break;

  case 427:
#line 994 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHRS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 428:
#line 995 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHRS" ); }
    break;

  case 429:
#line 999 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHLS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 430:
#line 1000 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHLS" ); }
    break;

  case 431:
#line 1004 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 432:
#line 1005 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVR" ); }
    break;

  case 433:
#line 1009 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 434:
#line 1010 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPR" ); }
    break;

  case 435:
#line 1014 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LSB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 436:
#line 1015 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LSB" ); }
    break;

  case 437:
#line 1019 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 438:
#line 1020 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 439:
#line 1021 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INDI" ); }
    break;

  case 440:
#line 1025 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 441:
#line 1026 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 442:
#line 1027 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "STEX" ); }
    break;

  case 443:
#line 1031 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_TRAC, (yyvsp[(2) - (2)]) ); }
    break;

  case 444:
#line 1032 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "TRAC" ); }
    break;

  case 445:
#line 1036 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_WRT, (yyvsp[(2) - (2)]) ); }
    break;

  case 446:
#line 1037 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "WRT" ); }
    break;

  case 447:
#line 1042 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STO, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 448:
#line 1043 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STO" ); }
    break;

  case 449:
#line 1047 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_FORB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 450:
#line 1048 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORB" ); }
    break;

  case 451:
#line 1052 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EVAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 452:
#line 1053 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EVAL" ); }
    break;

  case 453:
#line 1057 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed(true); COMPILER->addInstr( P_OOB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 454:
#line 1058 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "OOB" ); }
    break;


/* Line 1267 of yacc.c.  */
#line 4312 "/home/gian/Progetti/falcon/core/engine/fasm_parser.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 1061 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
 /* c code */


/****************************************************
* C Code for falcon HSM compiler
*****************************************************/


void fasm_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of falcon_parser.yxx */


