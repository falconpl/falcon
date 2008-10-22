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
     I_EVAL = 417
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




/* Copy the first part of user declarations.  */
#line 17 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"

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
#line 476 "/home/user/Progetti/falcon/core/engine/fasm_parser.cpp"

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
#define YYLAST   1796

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  163
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  123
/* YYNRULES -- Number of rules.  */
#define YYNRULES  451
/* YYNRULES -- Number of states.  */
#define YYNSTATES  818

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   417

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
     155,   156,   157,   158,   159,   160,   161,   162
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
     534,   536,   538,   540,   545,   548,   553,   556,   559,   562,
     567,   570,   575,   578,   583,   586,   591,   594,   599,   602,
     607,   610,   615,   618,   623,   626,   631,   634,   639,   642,
     647,   650,   655,   658,   663,   666,   671,   674,   679,   682,
     687,   690,   693,   696,   699,   702,   705,   708,   711,   714,
     717,   720,   723,   726,   729,   732,   735,   740,   745,   748,
     753,   758,   761,   766,   769,   774,   777,   780,   782,   785,
     788,   791,   794,   797,   800,   803,   806,   809,   814,   819,
     822,   829,   836,   839,   846,   853,   856,   863,   870,   873,
     878,   883,   886,   891,   896,   899,   906,   913,   916,   923,
     930,   933,   940,   947,   950,   955,   960,   963,   970,   977,
     980,   987,   994,   997,  1000,  1003,  1006,  1009,  1012,  1015,
    1018,  1021,  1024,  1031,  1034,  1037,  1040,  1043,  1046,  1049,
    1052,  1055,  1058,  1061,  1066,  1071,  1074,  1079,  1084,  1087,
    1092,  1097,  1100,  1103,  1106,  1109,  1111,  1114,  1116,  1119,
    1122,  1125,  1127,  1130,  1133,  1136,  1138,  1141,  1148,  1151,
    1158,  1161,  1165,  1171,  1176,  1181,  1184,  1189,  1194,  1199,
    1204,  1207,  1212,  1217,  1222,  1227,  1230,  1235,  1240,  1245,
    1250,  1253,  1256,  1259,  1262,  1267,  1270,  1275,  1278,  1283,
    1286,  1291,  1294,  1299,  1302,  1307,  1310,  1315,  1318,  1321,
    1324,  1329,  1334,  1337,  1342,  1347,  1350,  1355,  1360,  1363,
    1368,  1373,  1376,  1381,  1384,  1389,  1392,  1397,  1400,  1403,
    1406,  1409,  1412,  1417,  1420,  1425,  1428,  1433,  1436,  1441,
    1444,  1449,  1452,  1457,  1460,  1465,  1468,  1471,  1474,  1477,
    1480,  1483,  1486,  1489,  1492,  1495,  1498,  1503,  1506,  1511,
    1514,  1517
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     164,     0,    -1,    -1,   164,   165,    -1,     3,    -1,   175,
       3,    -1,   178,   180,     3,    -1,   178,     3,    -1,   180,
       3,    -1,     1,     3,    -1,    59,    -1,   167,    -1,   168,
      -1,   171,    -1,    42,    -1,   169,    -1,    46,    -1,    47,
      -1,    48,    -1,    49,    -1,    50,    -1,    51,    -1,    59,
      -1,   171,    -1,    52,    -1,    55,    -1,    56,    -1,   172,
      -1,    54,    -1,    53,    -1,    45,    -1,    54,    -1,    53,
      -1,    53,    -1,     4,    -1,     7,    -1,    26,     4,    -1,
       8,     4,   179,    -1,     8,     4,   179,    43,    -1,     9,
       4,   170,   179,    -1,     9,     4,   170,   179,    43,    -1,
      10,     4,   170,    -1,    10,     4,   170,    43,    -1,    11,
       4,   179,    -1,    11,     4,   179,    43,    -1,    12,     4,
     179,    -1,    13,     4,   179,    -1,    14,     4,    -1,    14,
       4,    43,    -1,    16,     4,   179,    -1,    16,     4,   179,
      43,    -1,    15,    -1,    27,     4,    -1,    27,    53,    -1,
      28,     4,   179,    -1,    28,     4,   179,     4,    -1,    28,
       4,   179,    53,    -1,    28,     4,   179,     4,   174,    -1,
      28,     4,   179,    53,   174,    -1,    29,     4,   179,    53,
     174,    -1,    29,     4,   179,     4,   174,    -1,    30,   168,
       5,     4,    -1,    30,   168,     5,    45,    -1,    31,   168,
       5,     4,    -1,    31,   168,     5,    45,    -1,    32,    59,
       5,     4,    -1,    32,    45,     5,     4,    -1,    32,    53,
       5,     4,    -1,    32,    54,     5,     4,    -1,    32,    42,
       5,     4,    -1,    32,    45,     6,    45,     5,     4,    -1,
      33,    -1,    18,     4,   170,    -1,    18,     4,    42,    -1,
      19,     4,    42,    -1,    38,   176,    -1,    39,   177,    -1,
      41,    42,     4,   179,    -1,    41,    42,     4,   179,    43,
      -1,    20,     4,   179,    -1,    20,     4,   179,    43,    -1,
      21,     4,    -1,    21,     4,    43,    -1,    22,    42,    -1,
      23,    -1,    40,    42,    -1,    24,     4,    -1,    25,     4,
     179,    -1,    34,    45,    -1,    35,    53,    -1,    36,    53,
      -1,    37,    53,    -1,    42,    -1,   176,     5,    42,    -1,
      42,    -1,   177,     5,    42,    -1,    44,     6,    -1,    -1,
      45,    -1,   181,    -1,   183,    -1,   184,    -1,   186,    -1,
     188,    -1,   190,    -1,   192,    -1,   193,    -1,   185,    -1,
     187,    -1,   189,    -1,   191,    -1,   201,    -1,   202,    -1,
     203,    -1,   204,    -1,   194,    -1,   195,    -1,   196,    -1,
     197,    -1,   198,    -1,   199,    -1,   205,    -1,   206,    -1,
     211,    -1,   212,    -1,   213,    -1,   215,    -1,   216,    -1,
     217,    -1,   218,    -1,   219,    -1,   220,    -1,   221,    -1,
     222,    -1,   223,    -1,   225,    -1,   224,    -1,   226,    -1,
     227,    -1,   228,    -1,   229,    -1,   230,    -1,   231,    -1,
     232,    -1,   233,    -1,   235,    -1,   236,    -1,   237,    -1,
     238,    -1,   239,    -1,   209,    -1,   210,    -1,   207,    -1,
     208,    -1,   241,    -1,   243,    -1,   242,    -1,   244,    -1,
     200,    -1,   245,    -1,   240,    -1,   234,    -1,   247,    -1,
     248,    -1,   182,    -1,   246,    -1,   251,    -1,   252,    -1,
     253,    -1,   254,    -1,   255,    -1,   256,    -1,   257,    -1,
     258,    -1,   259,    -1,   262,    -1,   260,    -1,   261,    -1,
     263,    -1,   264,    -1,   265,    -1,   266,    -1,   267,    -1,
     268,    -1,   269,    -1,   250,    -1,   214,    -1,   270,    -1,
     271,    -1,   272,    -1,   273,    -1,   274,    -1,   275,    -1,
     276,    -1,   277,    -1,   278,    -1,   279,    -1,   280,    -1,
     281,    -1,   282,    -1,   283,    -1,   284,    -1,   285,    -1,
      57,   168,     5,   166,    -1,    57,     1,    -1,   125,   168,
       5,   166,    -1,   125,     1,    -1,    58,   168,    -1,    58,
       1,    -1,    60,   166,     5,   166,    -1,    60,     1,    -1,
      66,   168,     5,   166,    -1,    66,     1,    -1,    61,   166,
       5,   166,    -1,    61,     1,    -1,    67,   168,     5,   166,
      -1,    67,     1,    -1,    62,   166,     5,   166,    -1,    62,
       1,    -1,    68,   168,     5,   166,    -1,    68,     1,    -1,
      63,   166,     5,   166,    -1,    63,     1,    -1,    69,   168,
       5,   166,    -1,    69,     1,    -1,    64,   166,     5,   166,
      -1,    64,     1,    -1,    65,   166,     5,   166,    -1,    65,
       1,    -1,   108,   166,     5,   166,    -1,   108,     1,    -1,
     109,   166,     5,   166,    -1,   109,     1,    -1,   111,   166,
       5,   166,    -1,   111,     1,    -1,   110,   166,     5,   166,
      -1,   110,     1,    -1,   113,   166,     5,   166,    -1,   113,
       1,    -1,   112,   166,     5,   166,    -1,   112,     1,    -1,
     121,     4,    -1,   121,    45,    -1,   121,     1,    -1,    71,
     168,    -1,    71,     1,    -1,    72,   168,    -1,    72,     1,
      -1,    73,   168,    -1,    73,     1,    -1,    74,   168,    -1,
      74,     1,    -1,    75,   166,    -1,    75,     1,    -1,    76,
     166,    -1,    76,     1,    -1,   116,    45,     5,   168,    -1,
     116,    45,     5,     4,    -1,   116,     1,    -1,   117,    45,
       5,   168,    -1,   117,    45,     5,     4,    -1,   117,     1,
      -1,   114,   168,     5,   168,    -1,   114,     1,    -1,   115,
      45,     5,   168,    -1,   115,     1,    -1,    81,   166,    -1,
      83,    -1,    81,     1,    -1,    82,   166,    -1,    82,     1,
      -1,    84,   168,    -1,    84,     1,    -1,   146,   168,    -1,
     146,     1,    -1,    99,   168,    -1,    99,     1,    -1,    85,
     168,     5,   168,    -1,    85,   168,     5,   172,    -1,    85,
       1,    -1,    86,   168,     5,   168,     5,   168,    -1,    86,
     168,     5,   172,     5,   168,    -1,    86,     1,    -1,    87,
     168,     5,   168,     5,   166,    -1,    87,   168,     5,   172,
       5,   166,    -1,    87,     1,    -1,    88,   168,     5,   168,
       5,   166,    -1,    88,   168,     5,   172,     5,   166,    -1,
      88,     1,    -1,    89,   168,     5,   168,    -1,    89,   168,
       5,   172,    -1,    89,     1,    -1,    90,   168,     5,   168,
      -1,    90,   168,     5,   172,    -1,    90,     1,    -1,    91,
     168,     5,   168,     5,   168,    -1,    91,   168,     5,   172,
       5,   168,    -1,    91,     1,    -1,    92,   168,     5,   168,
       5,   166,    -1,    92,   168,     5,   172,     5,   166,    -1,
      92,     1,    -1,    93,   168,     5,   168,     5,   168,    -1,
      93,   168,     5,   172,     5,   168,    -1,    93,     1,    -1,
      94,   168,     5,   168,    -1,    94,   168,     5,   172,    -1,
      94,     1,    -1,    95,    45,     5,   166,     5,   166,    -1,
      95,     4,     5,   166,     5,   166,    -1,    95,     1,    -1,
      96,     4,     5,     4,     5,    45,    -1,    96,    45,     5,
      45,     5,    45,    -1,    96,     1,    -1,    97,    45,    -1,
      97,     4,    -1,    97,     1,    -1,    98,    45,    -1,    98,
       1,    -1,   100,    45,    -1,   100,     1,    -1,   101,    45,
      -1,   101,     1,    -1,   102,   167,     5,   167,     5,   170,
      -1,   102,     1,    -1,   103,   167,    -1,   103,     1,    -1,
     124,   166,    -1,   124,     1,    -1,   104,     4,    -1,   104,
      45,    -1,   104,     1,    -1,   107,   166,    -1,   107,     1,
      -1,   105,     4,     5,   166,    -1,   105,    45,     5,   166,
      -1,   105,     1,    -1,   106,     4,     5,   166,    -1,   106,
      45,     5,   166,    -1,   106,     1,    -1,    80,    45,     5,
       4,    -1,    80,    45,     5,    45,    -1,    80,     1,    -1,
     122,     4,    -1,   122,    45,    -1,   122,     1,    -1,    77,
      -1,    77,     1,    -1,    79,    -1,    79,     1,    -1,    78,
     166,    -1,    78,     1,    -1,   120,    -1,   120,     1,    -1,
     123,    45,    -1,   123,     1,    -1,   145,    -1,   145,     1,
      -1,   118,    45,     5,   168,     5,   249,    -1,   118,     1,
      -1,   119,    45,     5,   168,     5,   249,    -1,   119,     1,
      -1,    45,     5,     4,    -1,   249,     5,    45,     5,     4,
      -1,   126,    45,     5,   166,    -1,   126,     4,     5,   166,
      -1,   126,     1,    -1,   127,    42,     5,    42,    -1,   127,
      42,     5,    45,    -1,   127,    45,     5,    42,    -1,   127,
      45,     5,    45,    -1,   127,     1,    -1,   128,    42,     5,
      42,    -1,   128,    42,     5,    45,    -1,   128,    45,     5,
      42,    -1,   128,    45,     5,    45,    -1,   128,     1,    -1,
     129,    42,     5,    42,    -1,   129,    42,     5,    45,    -1,
     129,    45,     5,    42,    -1,   129,    45,     5,    45,    -1,
     129,     1,    -1,   130,    42,    -1,   130,    45,    -1,   130,
       1,    -1,   132,   166,     5,   166,    -1,   132,     1,    -1,
     133,   166,     5,   166,    -1,   133,     1,    -1,   134,   168,
       5,   167,    -1,   134,     1,    -1,   135,   168,     5,   167,
      -1,   135,     1,    -1,   136,   168,     5,   167,    -1,   136,
       1,    -1,   131,   168,     5,   167,    -1,   131,     1,    -1,
      70,   168,     5,   167,    -1,    70,     1,    -1,   137,   168,
      -1,   137,     1,    -1,   138,   168,     5,   168,    -1,   138,
     168,     5,    45,    -1,   138,     1,    -1,   139,   168,     5,
     168,    -1,   139,   168,     5,    45,    -1,   139,     1,    -1,
     140,   168,     5,   168,    -1,   140,   168,     5,    45,    -1,
     140,     1,    -1,   141,   168,     5,   168,    -1,   141,   168,
       5,    45,    -1,   141,     1,    -1,   142,   166,     5,   167,
      -1,   142,     1,    -1,   143,   166,     5,   167,    -1,   143,
       1,    -1,   144,   166,     5,   167,    -1,   144,     1,    -1,
     147,   168,    -1,   147,     1,    -1,   148,   168,    -1,   148,
       1,    -1,   149,   166,     5,   166,    -1,   149,     1,    -1,
     150,   166,     5,   166,    -1,   150,     1,    -1,   151,   166,
       5,   166,    -1,   151,     1,    -1,   152,   166,     5,   166,
      -1,   152,     1,    -1,   153,   166,     5,   166,    -1,   153,
       1,    -1,   154,   166,     5,   166,    -1,   154,     1,    -1,
     155,   166,     5,   166,    -1,   155,     1,    -1,   156,   173,
      -1,   156,   168,    -1,   156,     1,    -1,   157,   173,    -1,
     157,   168,    -1,   157,     1,    -1,   158,   166,    -1,   158,
       1,    -1,   159,   166,    -1,   159,     1,    -1,   160,   168,
       5,   166,    -1,   160,     1,    -1,   161,   173,     5,   166,
      -1,   161,     1,    -1,   162,   166,    -1,   162,     1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   233,   233,   235,   239,   240,   241,   242,   243,   244,
     247,   247,   248,   248,   249,   249,   250,   250,   250,   250,
     250,   250,   252,   252,   253,   253,   253,   253,   254,   254,
     254,   255,   255,   256,   256,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   285,   288,
     291,   294,   295,   296,   297,   298,   299,   300,   301,   302,
     303,   304,   305,   306,   307,   308,   309,   310,   311,   312,
     313,   314,   315,   316,   317,   318,   319,   320,   321,   322,
     327,   333,   342,   343,   347,   348,   351,   355,   356,   360,
     361,   362,   363,   364,   365,   366,   367,   368,   369,   370,
     371,   372,   373,   374,   375,   376,   377,   378,   379,   380,
     381,   382,   383,   384,   385,   386,   387,   388,   389,   390,
     391,   392,   393,   394,   395,   396,   397,   398,   399,   400,
     401,   402,   403,   404,   405,   406,   407,   408,   409,   410,
     411,   412,   413,   414,   415,   416,   417,   418,   419,   420,
     421,   422,   423,   424,   425,   426,   427,   428,   429,   430,
     431,   432,   433,   434,   435,   436,   437,   438,   439,   440,
     441,   442,   443,   444,   445,   446,   447,   448,   449,   450,
     451,   452,   453,   454,   455,   456,   457,   458,   459,   460,
     461,   462,   463,   467,   468,   472,   473,   477,   478,   482,
     483,   487,   488,   493,   494,   498,   499,   503,   504,   508,
     509,   514,   515,   519,   520,   524,   525,   529,   530,   535,
     536,   540,   541,   545,   546,   550,   551,   555,   556,   560,
     561,   565,   566,   567,   571,   572,   576,   577,   582,   583,
     587,   588,   593,   594,   598,   599,   603,   604,   605,   609,
     610,   611,   615,   616,   620,   621,   626,   627,   628,   632,
     633,   638,   639,   643,   644,   648,   649,   654,   655,   656,
     660,   661,   662,   666,   667,   668,   672,   673,   674,   678,
     679,   680,   684,   685,   686,   690,   691,   692,   696,   697,
     698,   702,   703,   704,   708,   709,   710,   714,   715,   716,
     720,   721,   722,   726,   727,   728,   732,   733,   737,   738,
     742,   743,   747,   748,   752,   753,   757,   758,   762,   763,
     764,   768,   769,   773,   774,   775,   779,   780,   781,   786,
     787,   788,   792,   793,   794,   798,   799,   803,   804,   808,
     809,   813,   814,   818,   819,   823,   824,   828,   829,   833,
     834,   838,   847,   856,   857,   858,   862,   863,   864,   865,
     866,   870,   871,   872,   873,   874,   878,   879,   880,   881,
     882,   886,   887,   888,   892,   893,   897,   898,   902,   903,
     907,   908,   912,   913,   917,   918,   922,   923,   927,   928,
     932,   933,   934,   938,   939,   940,   944,   945,   946,   950,
     951,   952,   957,   958,   962,   963,   967,   968,   972,   973,
     977,   978,   982,   983,   987,   988,   992,   993,   997,   998,
    1002,  1003,  1007,  1008,  1012,  1013,  1017,  1018,  1019,  1023,
    1024,  1025,  1029,  1030,  1034,  1035,  1040,  1041,  1045,  1046,
    1050,  1051
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
  "I_TRAC", "I_WRT", "I_STO", "I_FORB", "I_EVAL", "$accept", "input",
  "line", "xoperand", "operand", "op_variable", "op_register",
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
  "inst_sto", "inst_forb", "inst_eval", 0
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
     415,   416,   417
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   163,   164,   164,   165,   165,   165,   165,   165,   165,
     166,   166,   167,   167,   168,   168,   169,   169,   169,   169,
     169,   169,   170,   170,   171,   171,   171,   171,   172,   172,
     172,   173,   173,   174,   174,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   176,   176,   177,   177,   178,   179,   179,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   181,   181,   182,   182,   183,   183,   184,
     184,   185,   185,   186,   186,   187,   187,   188,   188,   189,
     189,   190,   190,   191,   191,   192,   192,   193,   193,   194,
     194,   195,   195,   196,   196,   197,   197,   198,   198,   199,
     199,   200,   200,   200,   201,   201,   202,   202,   203,   203,
     204,   204,   205,   205,   206,   206,   207,   207,   207,   208,
     208,   208,   209,   209,   210,   210,   211,   211,   211,   212,
     212,   213,   213,   214,   214,   215,   215,   216,   216,   216,
     217,   217,   217,   218,   218,   218,   219,   219,   219,   220,
     220,   220,   221,   221,   221,   222,   222,   222,   223,   223,
     223,   224,   224,   224,   225,   225,   225,   226,   226,   226,
     227,   227,   227,   228,   228,   228,   229,   229,   230,   230,
     231,   231,   232,   232,   233,   233,   234,   234,   235,   235,
     235,   236,   236,   237,   237,   237,   238,   238,   238,   239,
     239,   239,   240,   240,   240,   241,   241,   242,   242,   243,
     243,   244,   244,   245,   245,   246,   246,   247,   247,   248,
     248,   249,   249,   250,   250,   250,   251,   251,   251,   251,
     251,   252,   252,   252,   252,   252,   253,   253,   253,   253,
     253,   254,   254,   254,   255,   255,   256,   256,   257,   257,
     258,   258,   259,   259,   260,   260,   261,   261,   262,   262,
     263,   263,   263,   264,   264,   264,   265,   265,   265,   266,
     266,   266,   267,   267,   268,   268,   269,   269,   270,   270,
     271,   271,   272,   272,   273,   273,   274,   274,   275,   275,
     276,   276,   277,   277,   278,   278,   279,   279,   279,   280,
     280,   280,   281,   281,   282,   282,   283,   283,   284,   284,
     285,   285
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
       1,     1,     1,     4,     2,     4,     2,     2,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     4,     4,     2,     4,
       4,     2,     4,     2,     4,     2,     2,     1,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     4,     4,     2,
       6,     6,     2,     6,     6,     2,     6,     6,     2,     4,
       4,     2,     4,     4,     2,     6,     6,     2,     6,     6,
       2,     6,     6,     2,     4,     4,     2,     6,     6,     2,
       6,     6,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     6,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     4,     4,     2,     4,     4,     2,     4,
       4,     2,     2,     2,     2,     1,     2,     1,     2,     2,
       2,     1,     2,     2,     2,     1,     2,     6,     2,     6,
       2,     3,     5,     4,     4,     2,     4,     4,     4,     4,
       2,     4,     4,     4,     4,     2,     4,     4,     4,     4,
       2,     2,     2,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     2,     2,
       4,     4,     2,     4,     4,     2,     4,     4,     2,     4,
       4,     2,     4,     2,     4,     2,     4,     2,     2,     2,
       2,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     4,     2,     4,     2,
       2,     2
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
       0,     0,     0,     0,     0,   267,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     3,     0,     0,     0,    99,
     164,   100,   101,   107,   102,   108,   103,   109,   104,   110,
     105,   106,   115,   116,   117,   118,   119,   120,   158,   111,
     112,   113,   114,   121,   122,   152,   153,   150,   151,   123,
     124,   125,   186,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   136,   135,   137,   138,   139,   140,   141,   142,
     143,   144,   161,   145,   146,   147,   148,   149,   160,   154,
     156,   155,   157,   159,   165,   162,   163,   185,   166,   167,
     168,   169,   170,   171,   172,   173,   174,   176,   177,   175,
     178,   179,   180,   181,   182,   183,   184,   187,   188,   189,
     190,   191,   192,   193,   194,   195,   196,   197,   198,   199,
     200,   201,   202,     9,    97,     0,     0,    97,    97,    97,
      47,    97,     0,     0,    97,    81,    83,    86,    97,    36,
      52,    53,    97,    97,    14,    16,    17,    18,    19,    20,
      21,     0,    15,     0,     0,     0,     0,     0,     0,    88,
      89,    90,    91,    92,    75,    94,    76,    85,     0,    96,
     204,     0,   208,   207,   210,    30,    24,    29,    28,    25,
      26,    10,     0,    11,    12,    13,    27,   214,     0,   218,
       0,   222,     0,   226,     0,   228,     0,   212,     0,   216,
       0,   220,     0,   224,     0,   397,     0,   245,   244,   247,
     246,   249,   248,   251,   250,   253,   252,   255,   254,   346,
     350,   349,   348,   341,     0,   268,   266,   270,   269,   272,
     271,   279,     0,   282,     0,   285,     0,   288,     0,   291,
       0,   294,     0,   297,     0,   300,     0,   303,     0,   306,
       0,   309,     0,     0,   312,     0,     0,   315,   314,   313,
     317,   316,   276,   275,   319,   318,   321,   320,   323,     0,
     325,   324,   330,   328,   329,   335,     0,     0,   338,     0,
       0,   332,   331,   230,     0,   232,     0,   236,     0,   234,
       0,   240,     0,   238,     0,   263,     0,   265,     0,   258,
       0,   261,     0,   358,     0,   360,     0,   352,   243,   241,
     242,   344,   342,   343,   354,   353,   327,   326,   206,     0,
     365,     0,     0,   370,     0,     0,   375,     0,     0,   380,
       0,     0,   383,   381,   382,   395,     0,   385,     0,   387,
       0,   389,     0,   391,     0,   393,     0,   399,   398,   402,
       0,   405,     0,   408,     0,   411,     0,   413,     0,   415,
       0,   417,     0,   356,   274,   273,   419,   418,   421,   420,
     423,     0,   425,     0,   427,     0,   429,     0,   431,     0,
     433,     0,   435,     0,   438,    32,    31,   437,   436,   441,
     440,   439,   443,   442,   445,   444,   447,     0,   449,     0,
     451,   450,     5,     7,     0,     8,    98,    37,    22,    97,
      23,    41,    43,    45,    46,    48,    49,    73,    72,    74,
      79,    82,    87,    54,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    97,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,    38,    39,    42,    44,    50,    80,    55,    56,
       0,     0,    61,    62,    63,    64,    69,    66,     0,    67,
      68,    65,    93,    95,    77,   203,   209,   213,   217,   221,
     225,   227,   211,   215,   219,   223,   396,   339,   340,   277,
     278,     0,     0,     0,     0,     0,     0,   289,   290,   292,
     293,     0,     0,     0,     0,     0,     0,   304,   305,     0,
       0,     0,     0,     0,   333,   334,   336,   337,   229,   231,
     235,   233,   239,   237,   262,   264,   257,   256,   260,   259,
       0,     0,   205,   364,   363,   366,   367,   368,   369,   371,
     372,   373,   374,   376,   377,   378,   379,   394,   384,   386,
     388,   390,   392,   401,   400,   404,   403,   407,   406,   410,
     409,   412,   414,   416,   422,   424,   426,   428,   430,   432,
     434,   446,   448,    40,    34,    33,    57,    58,    60,    59,
       0,    78,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    70,   280,   281,   283,   284,   286,   287,   295,   296,
     298,   299,   301,   302,   308,   307,   310,   311,   322,     0,
     357,   359,     0,     0,   361,     0,     0,   362
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,   145,   312,   313,   314,   282,   539,   315,   316,
     518,   766,   146,   294,   296,   147,   537,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   198,   199,   200,
     201,   202,   203,   204,   205,   206,   207,   208,   209,   210,
     211,   212,   213,   214,   215,   216,   810,   217,   218,   219,
     220,   221,   222,   223,   224,   225,   226,   227,   228,   229,
     230,   231,   232,   233,   234,   235,   236,   237,   238,   239,
     240,   241,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -473
static const yytype_int16 yypact[] =
{
    -473,   316,  -473,     8,  -473,  -473,    14,    31,    59,    69,
      79,   105,   137,  -473,   144,   155,   156,   168,   177,   157,
    -473,   196,   305,   306,     4,   307,   308,   466,   466,   439,
    -473,   268,   262,   269,   310,   322,   329,   330,   333,   474,
      96,   319,    81,   561,   664,   679,   721,   736,   542,   596,
    1336,  1350,  1360,  1378,  1388,  1402,  1412,   752,   808,   119,
     823,   181,    36,   838,   865,  -473,  1422,  1432,  1442,  1454,
    1464,  1474,  1484,  1494,  1506,  1516,  1526,    16,   113,   317,
      55,  1536,   313,   520,   478,  1321,   490,   499,   500,   880,
     895,   922,   937,   952,   979,   994,  1546,   538,   594,   595,
     604,   623,   187,   501,   505,   702,  1009,  1558,   506,   138,
     624,   625,   647,  1568,  1036,  1051,  1578,  1588,  1598,  1610,
    1620,  1630,  1640,  1650,  1066,  1093,  1108,   188,  1662,  1672,
    1682,  1123,  1150,  1165,  1180,  1207,  1222,  1237,   115,   521,
    1264,  1279,  1692,     6,  1294,  -473,   479,   146,   480,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,   440,   705,   705,   440,   440,   440,
     443,   440,   773,   445,   440,   446,  -473,  -473,   440,  -473,
    -473,  -473,   440,   440,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,   483,  -473,   485,   491,    93,   492,   494,   513,  -473,
    -473,  -473,  -473,  -473,   514,  -473,   559,  -473,   562,  -473,
    -473,   568,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,   571,  -473,  -473,  -473,  -473,  -473,   580,  -473,
     581,  -473,   582,  -473,   589,  -473,   597,  -473,   599,  -473,
     613,  -473,   614,  -473,   616,  -473,   617,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,   618,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,   648,  -473,   649,  -473,   669,  -473,   670,  -473,
     671,  -473,   672,  -473,   673,  -473,   685,  -473,   686,  -473,
     697,  -473,   699,   700,  -473,   703,   731,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,   734,
    -473,  -473,  -473,  -473,  -473,  -473,   735,   737,  -473,   738,
     739,  -473,  -473,  -473,   743,  -473,   744,  -473,   749,  -473,
     757,  -473,   760,  -473,   774,  -473,   788,  -473,   791,  -473,
     805,  -473,   807,  -473,   809,  -473,   811,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,   812,
    -473,   814,   815,  -473,   816,   817,  -473,   818,   825,  -473,
     826,   828,  -473,  -473,  -473,  -473,   829,  -473,   835,  -473,
     839,  -473,   842,  -473,   843,  -473,   844,  -473,  -473,  -473,
     846,  -473,   847,  -473,   900,  -473,   901,  -473,   903,  -473,
     904,  -473,   957,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,   958,  -473,   960,  -473,   961,  -473,  1014,  -473,  1015,
    -473,  1017,  -473,  1018,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  1071,  -473,  1072,
    -473,  -473,  -473,  -473,   704,  -473,  -473,   609,  -473,   440,
    -473,   698,   770,  -473,  -473,  -473,   800,  -473,  -473,  -473,
    1037,  -473,  -473,    11,    12,     5,    26,  1075,  1119,  1081,
    1124,  1125,  1126,  1085,  1089,   440,  1702,  1702,  1702,  1702,
    1702,  1702,  1702,  1702,  1702,  1702,  1702,  1717,    27,  1732,
    1732,  1732,  1732,  1732,  1732,  1732,  1732,  1732,  1732,  1702,
    1702,  1128,  1088,  1717,  1702,  1702,  1702,  1702,  1702,  1702,
    1702,  1702,  1702,  1702,   466,   466,    28,   637,   466,   466,
    1702,  1702,  1702,    -9,    42,    43,    68,   128,   129,  1717,
    1702,  1702,  1717,  1717,  1717,   853,   910,   967,  1024,  1717,
    1717,  1717,  1702,  1702,  1702,  1702,  1702,  1702,  1702,  1702,
    1702,  -473,  -473,  1091,  -473,  -473,  -473,  -473,    15,    15,
      15,    15,  -473,  -473,  -473,  -473,  -473,  -473,  1131,  -473,
    -473,  -473,  -473,  -473,  1094,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  1175,  1178,  1179,  1181,  1182,  1183,  -473,  -473,  -473,
    -473,  1184,  1185,  1186,  1188,  1189,  1232,  -473,  -473,  1235,
    1236,  1238,  1239,  1240,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    1241,  1242,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    1244,  -473,   466,   466,  1702,  1702,  1702,  1702,   466,   466,
    1702,  1702,   466,   466,  1702,  1702,  1140,  1197,   705,  1205,
    1205,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  1246,
    1289,  1289,  1293,  1253,  -473,  1295,  1297,  -473
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -473,  -473,  -473,    61,   -82,   -27,  -473,  -252,  -250,  1208,
    -105,  -472,  -473,  -473,  -473,  -473,  -206,  1152,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,   512,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,  -473,
    -473,  -473,  -473
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -356
static const yytype_int16 yytable[] =
{
     281,   283,   399,   401,   541,   540,   540,   528,   270,   652,
     548,   253,   540,   301,   303,   648,   650,   381,   254,   764,
     382,   328,   330,   332,   334,   336,   338,   340,   342,   344,
     654,   677,   716,   725,   521,   255,   726,   353,   529,   360,
     362,   364,   366,   368,   370,   372,   374,   376,   378,   380,
     653,   542,   543,   544,   393,   546,   390,   271,   550,   515,
     516,   383,   552,   256,   649,   651,   553,   554,   765,   426,
     274,   655,   678,   257,   275,   276,   277,   278,   279,   280,
     449,   354,   304,   258,   727,   729,   466,   728,   730,   472,
     474,   476,   478,   480,   482,   484,   486,   300,   558,   559,
     391,   495,   497,   499,   318,   320,   322,   324,   326,   259,
     731,   517,   520,   732,   384,   527,   514,   385,   346,   348,
     349,   351,  -345,   274,   356,   358,   305,   275,   276,   277,
     278,   279,   280,   306,   307,   308,   309,   310,   274,   453,
     311,   260,   275,   276,   277,   278,   279,   280,   261,   533,
     412,   414,   416,   418,   420,   422,   424,   274,   386,   262,
     263,   275,   276,   277,   278,   279,   280,   447,   515,   516,
     733,   735,   264,   734,   736,   468,   470,   767,   768,   769,
     454,   265,   352,   455,  -347,   488,   490,   492,   437,   493,
    -351,  -355,   501,   503,   505,   507,   509,   511,   513,   266,
     267,   523,   525,    40,    41,   531,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,   140,   141,   142,   143,   144,   268,
     269,   272,   273,   289,   394,   290,     2,     3,   387,     4,
     302,   388,   291,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,   643,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,   395,   664,
      39,   274,   389,   292,   293,   275,   276,   277,   278,   279,
     280,   295,   297,    40,    41,   298,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,   140,   141,   142,   143,   144,   398,
     299,   284,   532,   535,   285,   536,   545,   549,   555,   551,
     556,   402,   286,   287,   403,   676,   557,   560,   288,   561,
     405,   408,   438,   406,   409,   439,   441,   450,   274,   442,
     451,   703,   275,   276,   277,   278,   279,   280,   562,   563,
     274,   396,   519,   305,   275,   276,   277,   278,   279,   280,
     306,   307,   308,   309,   310,   404,   808,   737,   540,   427,
     740,   741,   742,   327,   407,   410,   440,   751,   752,   753,
     443,   452,   679,   681,   683,   685,   687,   689,   691,   693,
     695,   697,   317,   274,   564,   397,   565,   275,   276,   277,
     278,   279,   280,   566,   515,   516,   567,   714,   715,   717,
     719,   720,   721,   428,   274,   568,   569,   570,   275,   276,
     277,   278,   279,   280,   571,   429,   431,   329,   744,   746,
     748,   750,   572,   274,   573,   433,   305,   275,   276,   277,
     278,   279,   280,   306,   307,   308,   309,   310,   574,   575,
     311,   576,   577,   578,   435,   456,   459,   665,   666,   667,
     668,   669,   670,   671,   672,   673,   674,   675,   274,   430,
     432,   718,   275,   276,   277,   278,   279,   280,   462,   434,
     699,   700,   642,   579,   580,   704,   705,   706,   707,   708,
     709,   710,   711,   712,   713,   319,   457,   460,   436,   458,
     461,   722,   723,   724,   581,   582,   583,   584,   585,   274,
     321,   738,   739,   275,   276,   277,   278,   279,   280,   463,
     586,   587,   464,   754,   755,   756,   757,   758,   759,   760,
     761,   762,   588,   444,   589,   590,   274,   641,   591,   305,
     275,   276,   277,   278,   279,   280,   306,   307,   308,   309,
     310,   274,   323,   311,   305,   275,   276,   277,   278,   279,
     280,   306,   307,   308,   309,   310,   592,   325,   311,   593,
     594,   644,   595,   596,   597,   792,   793,   445,   598,   599,
     305,   798,   799,   345,   600,   802,   803,   306,   307,   308,
     309,   310,   601,   274,   538,   602,   305,   275,   276,   277,
     278,   279,   280,   306,   307,   308,   309,   310,   274,   603,
     311,   305,   275,   276,   277,   278,   279,   280,   306,   307,
     308,   309,   310,   604,   274,   311,   605,   305,   275,   276,
     277,   278,   279,   280,   306,   307,   308,   309,   310,   347,
     606,   311,   607,   645,   608,   547,   609,   610,   305,   611,
     612,   613,   614,   615,   350,   306,   307,   308,   309,   310,
     616,   617,   538,   618,   619,   794,   795,   796,   797,   355,
     620,   800,   801,   646,   621,   804,   805,   622,   623,   624,
     274,   625,   626,   305,   275,   276,   277,   278,   279,   280,
     306,   307,   308,   309,   310,   274,   357,   311,   305,   275,
     276,   277,   278,   279,   280,   306,   307,   308,   309,   310,
     274,   411,   311,   305,   275,   276,   277,   278,   279,   280,
     306,   307,   308,   309,   310,   274,   413,   311,   743,   275,
     276,   277,   278,   279,   280,   627,   628,   274,   629,   630,
     305,   275,   276,   277,   278,   279,   280,   306,   307,   308,
     309,   310,   274,   415,   311,   305,   275,   276,   277,   278,
     279,   280,   306,   307,   308,   309,   310,   274,   417,   311,
     305,   275,   276,   277,   278,   279,   280,   306,   307,   308,
     309,   310,   274,   419,   311,   745,   275,   276,   277,   278,
     279,   280,   631,   632,   274,   633,   634,   305,   275,   276,
     277,   278,   279,   280,   306,   307,   308,   309,   310,   274,
     421,   311,   305,   275,   276,   277,   278,   279,   280,   306,
     307,   308,   309,   310,   274,   423,   311,   305,   275,   276,
     277,   278,   279,   280,   306,   307,   308,   309,   310,   274,
     446,   311,   747,   275,   276,   277,   278,   279,   280,   635,
     636,   274,   637,   638,   305,   275,   276,   277,   278,   279,
     280,   306,   307,   308,   309,   310,   274,   467,   311,   305,
     275,   276,   277,   278,   279,   280,   306,   307,   308,   309,
     310,   274,   469,   311,   305,   275,   276,   277,   278,   279,
     280,   306,   307,   308,   309,   310,   274,   487,   311,   749,
     275,   276,   277,   278,   279,   280,   639,   640,   274,   656,
     647,   305,   275,   276,   277,   278,   279,   280,   306,   307,
     308,   309,   310,   274,   489,   311,   305,   275,   276,   277,
     278,   279,   280,   306,   307,   308,   309,   310,   274,   491,
     311,   305,   275,   276,   277,   278,   279,   280,   306,   307,
     308,   309,   310,   657,   500,   311,   658,   662,   659,   660,
     661,   663,   701,   702,   763,   274,   770,   771,   305,   275,
     276,   277,   278,   279,   280,   306,   307,   308,   309,   310,
     274,   502,   311,   305,   275,   276,   277,   278,   279,   280,
     306,   307,   308,   309,   310,   274,   504,   311,   305,   275,
     276,   277,   278,   279,   280,   306,   307,   308,   309,   310,
     772,   506,   311,   773,   774,   806,   775,   776,   777,   778,
     779,   780,   274,   781,   782,   305,   275,   276,   277,   278,
     279,   280,   306,   307,   308,   309,   310,   274,   508,   311,
     305,   275,   276,   277,   278,   279,   280,   306,   307,   308,
     309,   310,   274,   510,   311,   305,   275,   276,   277,   278,
     279,   280,   306,   307,   308,   309,   310,   783,   512,   311,
     784,   785,   807,   786,   787,   788,   789,   790,   791,   274,
     809,   812,   305,   275,   276,   277,   278,   279,   280,   306,
     307,   308,   309,   310,   274,   522,   311,   305,   275,   276,
     277,   278,   279,   280,   306,   307,   308,   309,   310,   274,
     524,   311,   305,   275,   276,   277,   278,   279,   280,   306,
     307,   308,   309,   310,   813,   530,   311,   814,   815,   534,
     816,   817,   811,     0,     0,     0,   274,     0,     0,   305,
     275,   276,   277,   278,   279,   280,   306,   307,   308,   309,
     310,   274,   400,   311,   305,   275,   276,   277,   278,   279,
     280,   306,   307,   308,   309,   310,   274,   331,   311,   305,
     275,   276,   277,   278,   279,   280,   306,   307,   308,   309,
     310,   333,     0,   311,     0,     0,     0,     0,     0,     0,
       0,   335,     0,   274,     0,     0,   305,   275,   276,   277,
     278,   279,   280,   306,   307,   308,   309,   310,   274,   337,
       0,     0,   275,   276,   277,   278,   279,   280,     0,   339,
       0,     0,   274,     0,     0,     0,   275,   276,   277,   278,
     279,   280,   274,   341,     0,     0,   275,   276,   277,   278,
     279,   280,     0,   343,     0,     0,     0,     0,     0,     0,
     274,     0,     0,   359,   275,   276,   277,   278,   279,   280,
     274,     0,     0,   361,   275,   276,   277,   278,   279,   280,
       0,     0,     0,   363,   274,     0,     0,     0,   275,   276,
     277,   278,   279,   280,   274,   365,     0,     0,   275,   276,
     277,   278,   279,   280,   274,   367,     0,     0,   275,   276,
     277,   278,   279,   280,   274,   369,     0,     0,   275,   276,
     277,   278,   279,   280,   274,   371,     0,     0,   275,   276,
     277,   278,   279,   280,     0,   373,   274,     0,     0,     0,
     275,   276,   277,   278,   279,   280,   274,   375,     0,     0,
     275,   276,   277,   278,   279,   280,   274,   377,     0,     0,
     275,   276,   277,   278,   279,   280,   274,   379,     0,     0,
     275,   276,   277,   278,   279,   280,   274,   392,     0,     0,
     275,   276,   277,   278,   279,   280,     0,   425,   274,     0,
       0,     0,   275,   276,   277,   278,   279,   280,   274,   448,
       0,     0,   275,   276,   277,   278,   279,   280,   274,   465,
       0,     0,   275,   276,   277,   278,   279,   280,   274,   471,
       0,     0,   275,   276,   277,   278,   279,   280,   274,   473,
       0,     0,   275,   276,   277,   278,   279,   280,     0,   475,
     274,     0,     0,     0,   275,   276,   277,   278,   279,   280,
     274,   477,     0,     0,   275,   276,   277,   278,   279,   280,
     274,   479,     0,     0,   275,   276,   277,   278,   279,   280,
     274,   481,     0,     0,   275,   276,   277,   278,   279,   280,
     274,   483,     0,     0,   275,   276,   277,   278,   279,   280,
       0,   485,   274,     0,     0,     0,   275,   276,   277,   278,
     279,   280,   274,   494,     0,     0,   275,   276,   277,   278,
     279,   280,   274,   496,     0,     0,   275,   276,   277,   278,
     279,   280,   274,   498,     0,     0,   275,   276,   277,   278,
     279,   280,   274,   526,     0,     0,   275,   276,   277,   278,
     279,   280,     0,     0,   274,     0,     0,     0,   275,   276,
     277,   278,   279,   280,   274,     0,     0,     0,   275,   276,
     277,   278,   279,   280,   274,     0,     0,     0,   275,   276,
     277,   278,   279,   280,   274,     0,     0,     0,   275,   276,
     277,   278,   279,   280,   274,     0,     0,   305,   275,   276,
     277,   278,   279,   280,   306,   307,   308,   309,   310,   274,
       0,   311,   305,   275,   276,   277,   278,   279,   280,   306,
     307,   308,   309,   310,   274,     0,     0,   305,   275,   276,
     277,   278,   279,   280,     0,   307,   308,   680,   682,   684,
     686,   688,   690,   692,   694,   696,   698
};

static const yytype_int16 yycheck[] =
{
      27,    28,    84,    85,   256,   255,   256,     1,     4,     4,
     262,     3,   262,    40,    41,     4,     4,     1,     4,     4,
       4,    48,    49,    50,    51,    52,    53,    54,    55,    56,
       4,     4,     4,    42,   139,     4,    45,     1,   143,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      45,   257,   258,   259,    81,   261,     1,    53,   264,    53,
      54,    45,   268,     4,    53,    53,   272,   273,    53,    96,
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
      42,    42,     4,    45,    45,   114,   115,   649,   650,   651,
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
     154,   155,   156,   157,   158,   159,   160,   161,   162,     4,
       4,     4,     4,    45,     1,    53,     0,     1,     1,     3,
       1,     4,    53,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,   539,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    45,   565,
      44,    42,    45,    53,    42,    46,    47,    48,    49,    50,
      51,    42,    42,    57,    58,    42,    60,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   147,   148,   149,   150,   151,   152,   153,
     154,   155,   156,   157,   158,   159,   160,   161,   162,     1,
       6,    42,     3,     3,    45,    45,    43,    42,     5,    43,
       5,     1,    53,    54,     4,   577,     5,     5,    59,     5,
       1,     1,     1,     4,     4,     4,     1,     1,    42,     4,
       4,   593,    46,    47,    48,    49,    50,    51,     5,     5,
      42,     1,     1,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    45,   788,   619,   788,     1,
     622,   623,   624,     1,    45,    45,    45,   629,   630,   631,
      45,    45,   579,   580,   581,   582,   583,   584,   585,   586,
     587,   588,     1,    42,     5,    45,     4,    46,    47,    48,
      49,    50,    51,     5,    53,    54,     5,   604,   605,   606,
     607,   608,   609,    45,    42,     5,     5,     5,    46,    47,
      48,    49,    50,    51,     5,     1,     1,     1,   625,   626,
     627,   628,     5,    42,     5,     1,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,     5,     5,
      59,     5,     5,     5,     1,     1,     1,   566,   567,   568,
     569,   570,   571,   572,   573,   574,   575,   576,    42,    45,
      45,     4,    46,    47,    48,    49,    50,    51,     1,    45,
     589,   590,    43,     5,     5,   594,   595,   596,   597,   598,
     599,   600,   601,   602,   603,     1,    42,    42,    45,    45,
      45,   610,   611,   612,     5,     5,     5,     5,     5,    42,
       1,   620,   621,    46,    47,    48,    49,    50,    51,    42,
       5,     5,    45,   632,   633,   634,   635,   636,   637,   638,
     639,   640,     5,     1,     5,     5,    42,     3,     5,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    42,     1,    59,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,     5,     1,    59,     5,
       5,    43,     5,     5,     5,   772,   773,    45,     5,     5,
      45,   778,   779,     1,     5,   782,   783,    52,    53,    54,
      55,    56,     5,    42,    59,     5,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    42,     5,
      59,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,     5,    42,    59,     5,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,     1,
       5,    59,     5,    43,     5,    42,     5,     5,    45,     5,
       5,     5,     5,     5,     1,    52,    53,    54,    55,    56,
       5,     5,    59,     5,     5,   774,   775,   776,   777,     1,
       5,   780,   781,    43,     5,   784,   785,     5,     5,     5,
      42,     5,     5,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    42,     1,    59,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      42,     1,    59,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    42,     1,    59,    45,    46,
      47,    48,    49,    50,    51,     5,     5,    42,     5,     5,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    42,     1,    59,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    42,     1,    59,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    42,     1,    59,    45,    46,    47,    48,    49,
      50,    51,     5,     5,    42,     5,     5,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    42,
       1,    59,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    42,     1,    59,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    42,
       1,    59,    45,    46,    47,    48,    49,    50,    51,     5,
       5,    42,     5,     5,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    42,     1,    59,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    42,     1,    59,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    42,     1,    59,    45,
      46,    47,    48,    49,    50,    51,     5,     5,    42,     4,
      43,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    42,     1,    59,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    42,     1,
      59,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,     4,     1,    59,    45,    42,     4,     4,
       4,    42,     4,    45,    43,    42,     5,    43,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      42,     1,    59,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    42,     1,    59,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
       5,     1,    59,     5,     5,    45,     5,     5,     5,     5,
       5,     5,    42,     5,     5,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    42,     1,    59,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    42,     1,    59,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,     5,     1,    59,
       5,     5,    45,     5,     5,     5,     5,     5,     4,    42,
      45,     5,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    42,     1,    59,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    42,
       1,    59,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,     5,     1,    59,     4,    45,   147,
       5,     4,   790,    -1,    -1,    -1,    42,    -1,    -1,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    42,     1,    59,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    42,     1,    59,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,     1,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    42,    -1,    -1,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    42,     1,
      -1,    -1,    46,    47,    48,    49,    50,    51,    -1,     1,
      -1,    -1,    42,    -1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    42,     1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,
      42,    -1,    -1,     1,    46,    47,    48,    49,    50,    51,
      42,    -1,    -1,     1,    46,    47,    48,    49,    50,    51,
      -1,    -1,    -1,     1,    42,    -1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    42,     1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    42,     1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    42,     1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    42,     1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    -1,     1,    42,    -1,    -1,    -1,
      46,    47,    48,    49,    50,    51,    42,     1,    -1,    -1,
      46,    47,    48,    49,    50,    51,    42,     1,    -1,    -1,
      46,    47,    48,    49,    50,    51,    42,     1,    -1,    -1,
      46,    47,    48,    49,    50,    51,    42,     1,    -1,    -1,
      46,    47,    48,    49,    50,    51,    -1,     1,    42,    -1,
      -1,    -1,    46,    47,    48,    49,    50,    51,    42,     1,
      -1,    -1,    46,    47,    48,    49,    50,    51,    42,     1,
      -1,    -1,    46,    47,    48,    49,    50,    51,    42,     1,
      -1,    -1,    46,    47,    48,    49,    50,    51,    42,     1,
      -1,    -1,    46,    47,    48,    49,    50,    51,    -1,     1,
      42,    -1,    -1,    -1,    46,    47,    48,    49,    50,    51,
      42,     1,    -1,    -1,    46,    47,    48,    49,    50,    51,
      42,     1,    -1,    -1,    46,    47,    48,    49,    50,    51,
      42,     1,    -1,    -1,    46,    47,    48,    49,    50,    51,
      42,     1,    -1,    -1,    46,    47,    48,    49,    50,    51,
      -1,     1,    42,    -1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    42,     1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    42,     1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    42,     1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    42,     1,    -1,    -1,    46,    47,    48,    49,
      50,    51,    -1,    -1,    42,    -1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    42,    -1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    42,    -1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    42,    -1,    -1,    -1,    46,    47,
      48,    49,    50,    51,    42,    -1,    -1,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    42,
      -1,    59,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    42,    -1,    -1,    45,    46,    47,
      48,    49,    50,    51,    -1,    53,    54,   579,   580,   581,
     582,   583,   584,   585,   586,   587,   588
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   164,     0,     1,     3,     7,     8,     9,    10,    11,
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
     158,   159,   160,   161,   162,   165,   175,   178,   180,   181,
     182,   183,   184,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   241,
     242,   243,   244,   245,   246,   247,   248,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,     3,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,    42,     4,     4,     4,
       4,    53,     4,     4,    42,    46,    47,    48,    49,    50,
      51,   168,   169,   168,    42,    45,    53,    54,    59,    45,
      53,    53,    53,    42,   176,    42,   177,    42,    42,     6,
       1,   168,     1,   168,     1,    45,    52,    53,    54,    55,
      56,    59,   166,   167,   168,   171,   172,     1,   166,     1,
     166,     1,   166,     1,   166,     1,   166,     1,   168,     1,
     168,     1,   168,     1,   168,     1,   168,     1,   168,     1,
     168,     1,   168,     1,   168,     1,   166,     1,   166,     1,
       1,   166,     1,     1,    45,     1,   166,     1,   166,     1,
     168,     1,   168,     1,   168,     1,   168,     1,   168,     1,
     168,     1,   168,     1,   168,     1,   168,     1,   168,     1,
     168,     1,     4,    45,     1,     4,    45,     1,     4,    45,
       1,    45,     1,   168,     1,    45,     1,    45,     1,   167,
       1,   167,     1,     4,    45,     1,     4,    45,     1,     4,
      45,     1,   166,     1,   166,     1,   166,     1,   166,     1,
     166,     1,   166,     1,   166,     1,   168,     1,    45,     1,
      45,     1,    45,     1,    45,     1,    45,     1,     1,     4,
      45,     1,     4,    45,     1,    45,     1,   166,     1,   168,
       1,     4,    45,     1,    42,    45,     1,    42,    45,     1,
      42,    45,     1,    42,    45,     1,   168,     1,   166,     1,
     166,     1,   168,     1,   168,     1,   168,     1,   168,     1,
     168,     1,   168,     1,   168,     1,   168,     1,   166,     1,
     166,     1,   166,     1,     1,   168,     1,   168,     1,   168,
       1,   166,     1,   166,     1,   166,     1,   166,     1,   166,
       1,   166,     1,   166,     1,    53,    54,   168,   173,     1,
     168,   173,     1,   166,     1,   166,     1,   168,     1,   173,
       1,   166,     3,     3,   180,     3,    45,   179,    59,   170,
     171,   170,   179,   179,   179,    43,   179,    42,   170,    42,
     179,    43,   179,   179,   179,     5,     5,     5,     5,     6,
       5,     5,     5,     5,     5,     4,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     3,    43,   179,    43,    43,    43,    43,     4,    53,
       4,    53,     4,    45,     4,    45,     4,     4,    45,     4,
       4,     4,    42,    42,   179,   166,   166,   166,   166,   166,
     166,   166,   166,   166,   166,   166,   167,     4,    45,   168,
     172,   168,   172,   168,   172,   168,   172,   168,   172,   168,
     172,   168,   172,   168,   172,   168,   172,   168,   172,   166,
     166,     4,    45,   167,   166,   166,   166,   166,   166,   166,
     166,   166,   166,   166,   168,   168,     4,   168,     4,   168,
     168,   168,   166,   166,   166,    42,    45,    42,    45,    42,
      45,    42,    45,    42,    45,    42,    45,   167,   166,   166,
     167,   167,   167,    45,   168,    45,   168,    45,   168,    45,
     168,   167,   167,   167,   166,   166,   166,   166,   166,   166,
     166,   166,   166,    43,     4,    53,   174,   174,   174,   174,
       5,    43,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     4,   168,   168,   166,   166,   166,   166,   168,   168,
     166,   166,   168,   168,   166,   166,    45,    45,   170,    45,
     249,   249,     5,     5,     4,    45,     5,     4
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
#line 244 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax, LINE - 1 ); }
    break;

  case 35:
#line 260 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addEntry(); }
    break;

  case 36:
#line 261 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->setModuleName( (yyvsp[(2) - (2)]) ); }
    break;

  case 37:
#line 262 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 38:
#line 263 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 39:
#line 264 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 40:
#line 265 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 41:
#line 266 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 42:
#line 267 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 43:
#line 268 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 44:
#line 269 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 45:
#line 270 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLocal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 46:
#line 271 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addParam( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 47:
#line 272 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 48:
#line 273 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 49:
#line 274 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 50:
#line 275 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 51:
#line 276 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); }
    break;

  case 52:
#line 277 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]), false ); }
    break;

  case 53:
#line 278 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]), true ); }
    break;

  case 54:
#line 279 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 55:
#line 280 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]), 0, false ); }
    break;

  case 56:
#line 281 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]), 0, true ); }
    break;

  case 57:
#line 282 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addImport( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), false );
   }
    break;

  case 58:
#line 285 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addImport( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), true );
   }
    break;

  case 59:
#line 288 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addAlias( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), true );
   }
    break;

  case 60:
#line 291 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addAlias( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), false );
   }
    break;

  case 61:
#line 294 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 62:
#line 295 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 63:
#line 296 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 64:
#line 297 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 65:
#line 298 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 66:
#line 299 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 67:
#line 300 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 68:
#line 301 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 69:
#line 302 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 70:
#line 303 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (6)]), (yyvsp[(6) - (6)]), (yyvsp[(4) - (6)]) ); }
    break;

  case 71:
#line 304 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDEndSwitch(); }
    break;

  case 72:
#line 305 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 73:
#line 306 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 74:
#line 307 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addPropRef( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 77:
#line 310 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 78:
#line 311 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 79:
#line 312 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 80:
#line 313 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 81:
#line 314 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 82:
#line 315 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 83:
#line 316 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassCtor( (yyvsp[(2) - (2)]) ); }
    break;

  case 84:
#line 317 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); /* Currently the same as .endfunc */ }
    break;

  case 85:
#line 318 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInherit((yyvsp[(2) - (2)])); }
    break;

  case 86:
#line 319 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFrom( (yyvsp[(2) - (2)]) ); }
    break;

  case 87:
#line 320 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addExtern( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 88:
#line 321 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addDLine( (yyvsp[(2) - (2)]) ); }
    break;

  case 89:
#line 323 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 90:
#line 328 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         (yyvsp[(2) - (2)])->asString().exported( true );
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 91:
#line 334 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 92:
#line 342 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(1) - (1)]) ); }
    break;

  case 93:
#line 343 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(3) - (3)]) ); }
    break;

  case 94:
#line 347 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(1) - (1)]) ); }
    break;

  case 95:
#line 348 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(3) - (3)]) ); }
    break;

  case 96:
#line 351 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->defineLabel( (yyvsp[(1) - (2)])->asLabel() ); }
    break;

  case 97:
#line 355 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {(yyval) = new Falcon::Pseudo( LINE, (Falcon::int64) 0 ); }
    break;

  case 203:
#line 467 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 204:
#line 468 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LD" ); }
    break;

  case 205:
#line 472 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDRF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 206:
#line 473 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDRF" ); }
    break;

  case 207:
#line 477 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LNIL, (yyvsp[(2) - (2)]) ); }
    break;

  case 208:
#line 478 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LNIL" ); }
    break;

  case 209:
#line 482 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 210:
#line 483 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADD" ); }
    break;

  case 211:
#line 487 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 212:
#line 488 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADDS" ); }
    break;

  case 213:
#line 493 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 214:
#line 494 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUB" ); }
    break;

  case 215:
#line 498 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUBS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 216:
#line 499 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUBS" ); }
    break;

  case 217:
#line 503 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MUL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 218:
#line 504 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MUL" ); }
    break;

  case 219:
#line 508 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MULS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 220:
#line 509 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MULS" ); }
    break;

  case 221:
#line 514 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 222:
#line 515 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIV" ); }
    break;

  case 223:
#line 519 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 224:
#line 520 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIVS" ); }
    break;

  case 225:
#line 524 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MOD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 226:
#line 525 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MOD" ); }
    break;

  case 227:
#line 529 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POW, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 228:
#line 530 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POW" ); }
    break;

  case 229:
#line 535 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 230:
#line 536 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EQ" ); }
    break;

  case 231:
#line 540 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 232:
#line 541 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEQ" ); }
    break;

  case 233:
#line 545 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 234:
#line 546 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GE" ); }
    break;

  case 235:
#line 550 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 236:
#line 551 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GT" ); }
    break;

  case 237:
#line 555 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 238:
#line 556 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LE" ); }
    break;

  case 239:
#line 560 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 240:
#line 561 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LT" ); }
    break;

  case 241:
#line 565 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 242:
#line 566 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 243:
#line 567 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRY" ); }
    break;

  case 244:
#line 571 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INC, (yyvsp[(2) - (2)]) ); }
    break;

  case 245:
#line 572 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INC" ); }
    break;

  case 246:
#line 576 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DEC, (yyvsp[(2) - (2)])  ); }
    break;

  case 247:
#line 577 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DEC" ); }
    break;

  case 248:
#line 582 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INCP, (yyvsp[(2) - (2)]) ); }
    break;

  case 249:
#line 583 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INCP" ); }
    break;

  case 250:
#line 587 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DECP, (yyvsp[(2) - (2)])  ); }
    break;

  case 251:
#line 588 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DECP" ); }
    break;

  case 252:
#line 593 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEG, (yyvsp[(2) - (2)])  ); }
    break;

  case 253:
#line 594 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEG" ); }
    break;

  case 254:
#line 598 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOT, (yyvsp[(2) - (2)])  ); }
    break;

  case 255:
#line 599 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOT" ); }
    break;

  case 256:
#line 603 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 257:
#line 604 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 258:
#line 605 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "CALL" ); }
    break;

  case 259:
#line 609 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 260:
#line 610 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 261:
#line 611 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INST" ); }
    break;

  case 262:
#line 615 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_UNPK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 263:
#line 616 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPK" ); }
    break;

  case 264:
#line 620 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_UNPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 265:
#line 621 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPS" ); }
    break;

  case 266:
#line 626 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PUSH, (yyvsp[(2) - (2)]) ); }
    break;

  case 267:
#line 627 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHN ); }
    break;

  case 268:
#line 628 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PUSH" ); }
    break;

  case 269:
#line 632 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHR, (yyvsp[(2) - (2)]) ); }
    break;

  case 270:
#line 633 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSHR" ); }
    break;

  case 271:
#line 638 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_POP, (yyvsp[(2) - (2)]) ); }
    break;

  case 272:
#line 639 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POP" ); }
    break;

  case 273:
#line 643 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PEEK, (yyvsp[(2) - (2)]) ); }
    break;

  case 274:
#line 644 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PEEK" ); }
    break;

  case 275:
#line 648 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 276:
#line 649 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XPOP" ); }
    break;

  case 277:
#line 654 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 278:
#line 655 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 279:
#line 656 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDV" ); }
    break;

  case 280:
#line 660 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 281:
#line 661 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 282:
#line 662 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVT" ); }
    break;

  case 283:
#line 666 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 284:
#line 667 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 285:
#line 668 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STV" ); }
    break;

  case 286:
#line 672 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 287:
#line 673 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 288:
#line 674 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVR" ); }
    break;

  case 289:
#line 678 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 290:
#line 679 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 291:
#line 680 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVS" ); }
    break;

  case 292:
#line 684 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 293:
#line 685 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 294:
#line 686 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDP" ); yyerrok; }
    break;

  case 295:
#line 690 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 296:
#line 691 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 297:
#line 692 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPT" ); yyerrok; }
    break;

  case 298:
#line 696 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 299:
#line 697 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 300:
#line 698 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STP" ); }
    break;

  case 301:
#line 702 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 302:
#line 703 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 303:
#line 704 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPR" ); }
    break;

  case 304:
#line 708 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 305:
#line 709 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 306:
#line 710 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPS" ); }
    break;

  case 307:
#line 714 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 308:
#line 715 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 309:
#line 716 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAV" ); }
    break;

  case 310:
#line 720 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 311:
#line 721 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 312:
#line 722 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAN" ); }
    break;

  case 313:
#line 726 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 314:
#line 727 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 315:
#line 728 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAL" ); }
    break;

  case 316:
#line 732 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_IPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 317:
#line 733 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IPOP" ); }
    break;

  case 318:
#line 737 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GENA, (yyvsp[(2) - (2)]) ); }
    break;

  case 319:
#line 738 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENA" ); }
    break;

  case 320:
#line 742 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GEND, (yyvsp[(2) - (2)]) ); }
    break;

  case 321:
#line 743 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEND" ); }
    break;

  case 322:
#line 747 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GENR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 323:
#line 748 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENR" ); }
    break;

  case 324:
#line 752 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GEOR, (yyvsp[(2) - (2)]) ); }
    break;

  case 325:
#line 753 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEOR" ); }
    break;

  case 326:
#line 757 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RIS, (yyvsp[(2) - (2)]) ); }
    break;

  case 327:
#line 758 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RIS" ); }
    break;

  case 328:
#line 762 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 329:
#line 763 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 330:
#line 764 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JMP" ); }
    break;

  case 331:
#line 768 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOOL, (yyvsp[(1) - (2)]) ); }
    break;

  case 332:
#line 769 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOOL" ); }
    break;

  case 333:
#line 773 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 334:
#line 774 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 335:
#line 775 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFT" ); }
    break;

  case 336:
#line 779 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 337:
#line 780 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 338:
#line 781 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFF" ); }
    break;

  case 339:
#line 786 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 340:
#line 787 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 341:
#line 788 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORK" ); }
    break;

  case 342:
#line 792 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 343:
#line 793 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 344:
#line 794 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JTRY" ); }
    break;

  case 345:
#line 798 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RET ); }
    break;

  case 346:
#line 799 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RET" ); }
    break;

  case 347:
#line 803 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETA ); }
    break;

  case 348:
#line 804 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETA" ); }
    break;

  case 349:
#line 808 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETV, (yyvsp[(2) - (2)]) ); }
    break;

  case 350:
#line 809 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETV" ); }
    break;

  case 351:
#line 813 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOP ); }
    break;

  case 352:
#line 814 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOP" ); }
    break;

  case 353:
#line 818 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_PTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 354:
#line 819 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PTRY" ); }
    break;

  case 355:
#line 823 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_END ); }
    break;

  case 356:
#line 824 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "END" ); }
    break;

  case 357:
#line 828 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 358:
#line 829 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SWCH" ); }
    break;

  case 359:
#line 833 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 360:
#line 834 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SELE" ); }
    break;

  case 361:
#line 839 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         Falcon::Pseudo *psd = new Falcon::Pseudo( Falcon::Pseudo::tswitch_list );
         psd->line( LINE );
         psd->asList()->pushBack( (yyvsp[(1) - (3)]) );
         psd->asList()->pushBack( (yyvsp[(3) - (3)]) );
         (yyval) = psd;
      }
    break;

  case 362:
#line 848 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(3) - (5)]) );
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(5) - (5)]) );
         (yyval) = (yyvsp[(1) - (5)]);
      }
    break;

  case 363:
#line 856 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 364:
#line 857 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 365:
#line 858 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ONCE" ); }
    break;

  case 366:
#line 862 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 367:
#line 863 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 368:
#line 864 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 369:
#line 865 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 370:
#line 866 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BAND" ); }
    break;

  case 371:
#line 870 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 372:
#line 871 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 373:
#line 872 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 374:
#line 873 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 375:
#line 874 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOR" ); }
    break;

  case 376:
#line 878 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 377:
#line 879 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 378:
#line 880 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 379:
#line 881 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 380:
#line 882 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 381:
#line 886 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 382:
#line 887 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 383:
#line 888 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 384:
#line 892 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_AND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 385:
#line 893 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "AND" ); }
    break;

  case 386:
#line 897 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_OR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 387:
#line 898 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "OR" ); }
    break;

  case 388:
#line 902 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ANDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 389:
#line 903 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ANDS" ); }
    break;

  case 390:
#line 907 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 391:
#line 908 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ORS" ); }
    break;

  case 392:
#line 912 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 393:
#line 913 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XORS" ); }
    break;

  case 394:
#line 917 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MODS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 395:
#line 918 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MODS" ); }
    break;

  case 396:
#line 922 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POWS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 397:
#line 923 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POWS" ); }
    break;

  case 398:
#line 927 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOTS, (yyvsp[(2) - (2)]) ); }
    break;

  case 399:
#line 928 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOTS" ); }
    break;

  case 400:
#line 932 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 401:
#line 933 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 402:
#line 934 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HAS" ); }
    break;

  case 403:
#line 938 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 404:
#line 939 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 405:
#line 940 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HASN" ); }
    break;

  case 406:
#line 944 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 407:
#line 945 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 408:
#line 946 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVE" ); }
    break;

  case 409:
#line 950 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 410:
#line 951 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 411:
#line 952 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVN" ); }
    break;

  case 412:
#line 957 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_IN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 413:
#line 958 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IN" ); }
    break;

  case 414:
#line 962 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOIN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 415:
#line 963 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOIN" ); }
    break;

  case 416:
#line 967 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PROV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 417:
#line 968 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PROV" ); }
    break;

  case 418:
#line 972 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSIN, (yyvsp[(2) - (2)]) ); }
    break;

  case 419:
#line 973 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSIN" ); }
    break;

  case 420:
#line 977 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PASS, (yyvsp[(2) - (2)]) ); }
    break;

  case 421:
#line 978 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PASS" ); }
    break;

  case 422:
#line 982 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 423:
#line 983 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHR" ); }
    break;

  case 424:
#line 987 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 425:
#line 988 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHL" ); }
    break;

  case 426:
#line 992 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHRS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 427:
#line 993 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHRS" ); }
    break;

  case 428:
#line 997 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHLS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 429:
#line 998 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHLS" ); }
    break;

  case 430:
#line 1002 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 431:
#line 1003 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVR" ); }
    break;

  case 432:
#line 1007 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 433:
#line 1008 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPR" ); }
    break;

  case 434:
#line 1012 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LSB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 435:
#line 1013 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LSB" ); }
    break;

  case 436:
#line 1017 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 437:
#line 1018 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 438:
#line 1019 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INDI" ); }
    break;

  case 439:
#line 1023 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 440:
#line 1024 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 441:
#line 1025 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "STEX" ); }
    break;

  case 442:
#line 1029 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_TRAC, (yyvsp[(2) - (2)]) ); }
    break;

  case 443:
#line 1030 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "TRAC" ); }
    break;

  case 444:
#line 1034 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_WRT, (yyvsp[(2) - (2)]) ); }
    break;

  case 445:
#line 1035 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "WRT" ); }
    break;

  case 446:
#line 1040 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STO, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 447:
#line 1041 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STO" ); }
    break;

  case 448:
#line 1045 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_FORB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 449:
#line 1046 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORB" ); }
    break;

  case 450:
#line 1050 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EVAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 451:
#line 1051 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EVAL" ); }
    break;


/* Line 1267 of yacc.c.  */
#line 4272 "/home/user/Progetti/falcon/core/engine/fasm_parser.cpp"
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


#line 1054 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
 /* c code */


/****************************************************
* C Code for falcon HSM compiler
*****************************************************/


void fasm_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of falcon_parser.yxx */


