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
     DSWITCH = 284,
     DSELECT = 285,
     DCASE = 286,
     DENDSWITCH = 287,
     DLINE = 288,
     DSTRING = 289,
     DISTRING = 290,
     DCSTRING = 291,
     DHAS = 292,
     DHASNT = 293,
     DINHERIT = 294,
     DINSTANCE = 295,
     SYMBOL = 296,
     EXPORT = 297,
     LABEL = 298,
     INTEGER = 299,
     REG_A = 300,
     REG_B = 301,
     REG_S1 = 302,
     REG_S2 = 303,
     REG_L1 = 304,
     REG_L2 = 305,
     NUMERIC = 306,
     STRING = 307,
     STRING_ID = 308,
     TRUE_TOKEN = 309,
     FALSE_TOKEN = 310,
     I_LD = 311,
     I_LNIL = 312,
     NIL = 313,
     I_ADD = 314,
     I_SUB = 315,
     I_MUL = 316,
     I_DIV = 317,
     I_MOD = 318,
     I_POW = 319,
     I_ADDS = 320,
     I_SUBS = 321,
     I_MULS = 322,
     I_DIVS = 323,
     I_POWS = 324,
     I_INC = 325,
     I_DEC = 326,
     I_INCP = 327,
     I_DECP = 328,
     I_NEG = 329,
     I_NOT = 330,
     I_RET = 331,
     I_RETV = 332,
     I_RETA = 333,
     I_FORK = 334,
     I_PUSH = 335,
     I_PSHR = 336,
     I_PSHN = 337,
     I_POP = 338,
     I_LDV = 339,
     I_LDVT = 340,
     I_STV = 341,
     I_STVR = 342,
     I_STVS = 343,
     I_LDP = 344,
     I_LDPT = 345,
     I_STP = 346,
     I_STPR = 347,
     I_STPS = 348,
     I_TRAV = 349,
     I_TRAN = 350,
     I_TRAL = 351,
     I_IPOP = 352,
     I_XPOP = 353,
     I_GENA = 354,
     I_GEND = 355,
     I_GENR = 356,
     I_GEOR = 357,
     I_JMP = 358,
     I_IFT = 359,
     I_IFF = 360,
     I_BOOL = 361,
     I_EQ = 362,
     I_NEQ = 363,
     I_GT = 364,
     I_GE = 365,
     I_LT = 366,
     I_LE = 367,
     I_UNPK = 368,
     I_UNPS = 369,
     I_CALL = 370,
     I_INST = 371,
     I_SWCH = 372,
     I_SELE = 373,
     I_NOP = 374,
     I_TRY = 375,
     I_JTRY = 376,
     I_PTRY = 377,
     I_RIS = 378,
     I_LDRF = 379,
     I_ONCE = 380,
     I_BAND = 381,
     I_BOR = 382,
     I_BXOR = 383,
     I_BNOT = 384,
     I_MODS = 385,
     I_AND = 386,
     I_OR = 387,
     I_ANDS = 388,
     I_ORS = 389,
     I_XORS = 390,
     I_NOTS = 391,
     I_HAS = 392,
     I_HASN = 393,
     I_GIVE = 394,
     I_GIVN = 395,
     I_IN = 396,
     I_NOIN = 397,
     I_PROV = 398,
     I_END = 399,
     I_PEEK = 400,
     I_PSIN = 401,
     I_PASS = 402,
     I_SHR = 403,
     I_SHL = 404,
     I_SHRS = 405,
     I_SHLS = 406,
     I_LDVR = 407,
     I_LDPR = 408,
     I_LSB = 409,
     I_INDI = 410,
     I_STEX = 411,
     I_TRAC = 412,
     I_WRT = 413,
     I_STO = 414,
     I_FORB = 415
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
#define DSWITCH 284
#define DSELECT 285
#define DCASE 286
#define DENDSWITCH 287
#define DLINE 288
#define DSTRING 289
#define DISTRING 290
#define DCSTRING 291
#define DHAS 292
#define DHASNT 293
#define DINHERIT 294
#define DINSTANCE 295
#define SYMBOL 296
#define EXPORT 297
#define LABEL 298
#define INTEGER 299
#define REG_A 300
#define REG_B 301
#define REG_S1 302
#define REG_S2 303
#define REG_L1 304
#define REG_L2 305
#define NUMERIC 306
#define STRING 307
#define STRING_ID 308
#define TRUE_TOKEN 309
#define FALSE_TOKEN 310
#define I_LD 311
#define I_LNIL 312
#define NIL 313
#define I_ADD 314
#define I_SUB 315
#define I_MUL 316
#define I_DIV 317
#define I_MOD 318
#define I_POW 319
#define I_ADDS 320
#define I_SUBS 321
#define I_MULS 322
#define I_DIVS 323
#define I_POWS 324
#define I_INC 325
#define I_DEC 326
#define I_INCP 327
#define I_DECP 328
#define I_NEG 329
#define I_NOT 330
#define I_RET 331
#define I_RETV 332
#define I_RETA 333
#define I_FORK 334
#define I_PUSH 335
#define I_PSHR 336
#define I_PSHN 337
#define I_POP 338
#define I_LDV 339
#define I_LDVT 340
#define I_STV 341
#define I_STVR 342
#define I_STVS 343
#define I_LDP 344
#define I_LDPT 345
#define I_STP 346
#define I_STPR 347
#define I_STPS 348
#define I_TRAV 349
#define I_TRAN 350
#define I_TRAL 351
#define I_IPOP 352
#define I_XPOP 353
#define I_GENA 354
#define I_GEND 355
#define I_GENR 356
#define I_GEOR 357
#define I_JMP 358
#define I_IFT 359
#define I_IFF 360
#define I_BOOL 361
#define I_EQ 362
#define I_NEQ 363
#define I_GT 364
#define I_GE 365
#define I_LT 366
#define I_LE 367
#define I_UNPK 368
#define I_UNPS 369
#define I_CALL 370
#define I_INST 371
#define I_SWCH 372
#define I_SELE 373
#define I_NOP 374
#define I_TRY 375
#define I_JTRY 376
#define I_PTRY 377
#define I_RIS 378
#define I_LDRF 379
#define I_ONCE 380
#define I_BAND 381
#define I_BOR 382
#define I_BXOR 383
#define I_BNOT 384
#define I_MODS 385
#define I_AND 386
#define I_OR 387
#define I_ANDS 388
#define I_ORS 389
#define I_XORS 390
#define I_NOTS 391
#define I_HAS 392
#define I_HASN 393
#define I_GIVE 394
#define I_GIVN 395
#define I_IN 396
#define I_NOIN 397
#define I_PROV 398
#define I_END 399
#define I_PEEK 400
#define I_PSIN 401
#define I_PASS 402
#define I_SHR 403
#define I_SHL 404
#define I_SHRS 405
#define I_SHLS 406
#define I_LDVR 407
#define I_LDPR 408
#define I_LSB 409
#define I_INDI 410
#define I_STEX 411
#define I_TRAC 412
#define I_WRT 413
#define I_STO 414
#define I_FORB 415




/* Copy the first part of user declarations.  */
#line 17 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"

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
#line 472 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.cpp"

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
#define YYLAST   1826

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  161
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  122
/* YYNRULES -- Number of rules.  */
#define YYNRULES  446
/* YYNRULES -- Number of states.  */
#define YYNSTATES  807

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   415

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
     155,   156,   157,   158,   159,   160
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
     181,   186,   191,   196,   201,   206,   211,   216,   221,   228,
     230,   234,   238,   242,   245,   248,   253,   259,   263,   268,
     271,   275,   278,   280,   283,   286,   290,   293,   296,   299,
     302,   304,   308,   310,   314,   317,   318,   320,   322,   324,
     326,   328,   330,   332,   334,   336,   338,   340,   342,   344,
     346,   348,   350,   352,   354,   356,   358,   360,   362,   364,
     366,   368,   370,   372,   374,   376,   378,   380,   382,   384,
     386,   388,   390,   392,   394,   396,   398,   400,   402,   404,
     406,   408,   410,   412,   414,   416,   418,   420,   422,   424,
     426,   428,   430,   432,   434,   436,   438,   440,   442,   444,
     446,   448,   450,   452,   454,   456,   458,   460,   462,   464,
     466,   468,   470,   472,   474,   476,   478,   480,   482,   484,
     486,   488,   490,   492,   494,   496,   498,   500,   502,   504,
     506,   508,   510,   512,   514,   516,   518,   520,   522,   524,
     526,   531,   534,   539,   542,   545,   548,   553,   556,   561,
     564,   569,   572,   577,   580,   585,   588,   593,   596,   601,
     604,   609,   612,   617,   620,   625,   628,   633,   636,   641,
     644,   649,   652,   657,   660,   665,   668,   673,   676,   679,
     682,   685,   688,   691,   694,   697,   700,   703,   706,   709,
     712,   715,   718,   721,   726,   731,   734,   739,   744,   747,
     752,   755,   760,   763,   766,   768,   771,   774,   777,   780,
     783,   786,   789,   792,   795,   800,   805,   808,   815,   822,
     825,   832,   839,   842,   849,   856,   859,   864,   869,   872,
     877,   882,   885,   892,   899,   902,   909,   916,   919,   926,
     933,   936,   941,   946,   949,   956,   963,   966,   973,   980,
     983,   986,   989,   992,   995,   998,  1001,  1004,  1007,  1010,
    1017,  1020,  1023,  1026,  1029,  1032,  1035,  1038,  1041,  1044,
    1047,  1052,  1057,  1060,  1065,  1070,  1073,  1078,  1083,  1086,
    1089,  1092,  1095,  1097,  1100,  1102,  1105,  1108,  1111,  1113,
    1116,  1119,  1122,  1124,  1127,  1134,  1137,  1144,  1147,  1151,
    1157,  1162,  1167,  1170,  1175,  1180,  1185,  1190,  1193,  1198,
    1203,  1208,  1213,  1216,  1221,  1226,  1231,  1236,  1239,  1242,
    1245,  1248,  1253,  1256,  1261,  1264,  1269,  1272,  1277,  1280,
    1285,  1288,  1293,  1296,  1301,  1304,  1307,  1310,  1315,  1320,
    1323,  1328,  1333,  1336,  1341,  1346,  1349,  1354,  1359,  1362,
    1367,  1370,  1375,  1378,  1383,  1386,  1389,  1392,  1395,  1398,
    1403,  1406,  1411,  1414,  1419,  1422,  1427,  1430,  1435,  1438,
    1443,  1446,  1451,  1454,  1457,  1460,  1463,  1466,  1469,  1472,
    1475,  1478,  1481,  1484,  1489,  1492,  1497
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     162,     0,    -1,    -1,   162,   163,    -1,     3,    -1,   173,
       3,    -1,   176,   178,     3,    -1,   176,     3,    -1,   178,
       3,    -1,     1,     3,    -1,    58,    -1,   165,    -1,   166,
      -1,   169,    -1,    41,    -1,   167,    -1,    45,    -1,    46,
      -1,    47,    -1,    48,    -1,    49,    -1,    50,    -1,    58,
      -1,   169,    -1,    51,    -1,    54,    -1,    55,    -1,   170,
      -1,    53,    -1,    52,    -1,    44,    -1,    53,    -1,    52,
      -1,    52,    -1,     4,    -1,     7,    -1,    26,     4,    -1,
       8,     4,   177,    -1,     8,     4,   177,    42,    -1,     9,
       4,   168,   177,    -1,     9,     4,   168,   177,    42,    -1,
      10,     4,   168,    -1,    10,     4,   168,    42,    -1,    11,
       4,   177,    -1,    11,     4,   177,    42,    -1,    12,     4,
     177,    -1,    13,     4,   177,    -1,    14,     4,    -1,    14,
       4,    42,    -1,    16,     4,   177,    -1,    16,     4,   177,
      42,    -1,    15,    -1,    27,     4,    -1,    27,    52,    -1,
      28,     4,   177,    -1,    28,     4,   177,     4,    -1,    28,
       4,   177,    52,    -1,    28,     4,   177,     4,   172,    -1,
      28,     4,   177,    52,   172,    -1,    29,   166,     5,     4,
      -1,    29,   166,     5,    44,    -1,    30,   166,     5,     4,
      -1,    30,   166,     5,    44,    -1,    31,    58,     5,     4,
      -1,    31,    44,     5,     4,    -1,    31,    52,     5,     4,
      -1,    31,    53,     5,     4,    -1,    31,    41,     5,     4,
      -1,    31,    44,     6,    44,     5,     4,    -1,    32,    -1,
      18,     4,   168,    -1,    18,     4,    41,    -1,    19,     4,
      41,    -1,    37,   174,    -1,    38,   175,    -1,    40,    41,
       4,   177,    -1,    40,    41,     4,   177,    42,    -1,    20,
       4,   177,    -1,    20,     4,   177,    42,    -1,    21,     4,
      -1,    21,     4,    42,    -1,    22,    41,    -1,    23,    -1,
      39,    41,    -1,    24,     4,    -1,    25,     4,   177,    -1,
      33,    44,    -1,    34,    52,    -1,    35,    52,    -1,    36,
      52,    -1,    41,    -1,   174,     5,    41,    -1,    41,    -1,
     175,     5,    41,    -1,    43,     6,    -1,    -1,    44,    -1,
     179,    -1,   181,    -1,   182,    -1,   184,    -1,   186,    -1,
     188,    -1,   190,    -1,   191,    -1,   183,    -1,   185,    -1,
     187,    -1,   189,    -1,   199,    -1,   200,    -1,   201,    -1,
     202,    -1,   192,    -1,   193,    -1,   194,    -1,   195,    -1,
     196,    -1,   197,    -1,   203,    -1,   204,    -1,   209,    -1,
     210,    -1,   211,    -1,   213,    -1,   214,    -1,   215,    -1,
     216,    -1,   217,    -1,   218,    -1,   219,    -1,   220,    -1,
     221,    -1,   223,    -1,   222,    -1,   224,    -1,   225,    -1,
     226,    -1,   227,    -1,   228,    -1,   229,    -1,   230,    -1,
     231,    -1,   233,    -1,   234,    -1,   235,    -1,   236,    -1,
     237,    -1,   207,    -1,   208,    -1,   205,    -1,   206,    -1,
     239,    -1,   241,    -1,   240,    -1,   242,    -1,   198,    -1,
     243,    -1,   238,    -1,   232,    -1,   245,    -1,   246,    -1,
     180,    -1,   244,    -1,   249,    -1,   250,    -1,   251,    -1,
     252,    -1,   253,    -1,   254,    -1,   255,    -1,   256,    -1,
     257,    -1,   260,    -1,   258,    -1,   259,    -1,   261,    -1,
     262,    -1,   263,    -1,   264,    -1,   265,    -1,   266,    -1,
     267,    -1,   248,    -1,   212,    -1,   268,    -1,   269,    -1,
     270,    -1,   271,    -1,   272,    -1,   273,    -1,   274,    -1,
     275,    -1,   276,    -1,   277,    -1,   278,    -1,   279,    -1,
     280,    -1,   281,    -1,   282,    -1,    56,   166,     5,   164,
      -1,    56,     1,    -1,   124,   166,     5,   164,    -1,   124,
       1,    -1,    57,   166,    -1,    57,     1,    -1,    59,   164,
       5,   164,    -1,    59,     1,    -1,    65,   166,     5,   164,
      -1,    65,     1,    -1,    60,   164,     5,   164,    -1,    60,
       1,    -1,    66,   166,     5,   164,    -1,    66,     1,    -1,
      61,   164,     5,   164,    -1,    61,     1,    -1,    67,   166,
       5,   164,    -1,    67,     1,    -1,    62,   164,     5,   164,
      -1,    62,     1,    -1,    68,   166,     5,   164,    -1,    68,
       1,    -1,    63,   164,     5,   164,    -1,    63,     1,    -1,
      64,   164,     5,   164,    -1,    64,     1,    -1,   107,   164,
       5,   164,    -1,   107,     1,    -1,   108,   164,     5,   164,
      -1,   108,     1,    -1,   110,   164,     5,   164,    -1,   110,
       1,    -1,   109,   164,     5,   164,    -1,   109,     1,    -1,
     112,   164,     5,   164,    -1,   112,     1,    -1,   111,   164,
       5,   164,    -1,   111,     1,    -1,   120,     4,    -1,   120,
      44,    -1,   120,     1,    -1,    70,   166,    -1,    70,     1,
      -1,    71,   166,    -1,    71,     1,    -1,    72,   166,    -1,
      72,     1,    -1,    73,   166,    -1,    73,     1,    -1,    74,
     164,    -1,    74,     1,    -1,    75,   164,    -1,    75,     1,
      -1,   115,    44,     5,   166,    -1,   115,    44,     5,     4,
      -1,   115,     1,    -1,   116,    44,     5,   166,    -1,   116,
      44,     5,     4,    -1,   116,     1,    -1,   113,   166,     5,
     166,    -1,   113,     1,    -1,   114,    44,     5,   166,    -1,
     114,     1,    -1,    80,   164,    -1,    82,    -1,    80,     1,
      -1,    81,   164,    -1,    81,     1,    -1,    83,   166,    -1,
      83,     1,    -1,   145,   166,    -1,   145,     1,    -1,    98,
     166,    -1,    98,     1,    -1,    84,   166,     5,   166,    -1,
      84,   166,     5,   170,    -1,    84,     1,    -1,    85,   166,
       5,   166,     5,   166,    -1,    85,   166,     5,   170,     5,
     166,    -1,    85,     1,    -1,    86,   166,     5,   166,     5,
     164,    -1,    86,   166,     5,   170,     5,   164,    -1,    86,
       1,    -1,    87,   166,     5,   166,     5,   164,    -1,    87,
     166,     5,   170,     5,   164,    -1,    87,     1,    -1,    88,
     166,     5,   166,    -1,    88,   166,     5,   170,    -1,    88,
       1,    -1,    89,   166,     5,   166,    -1,    89,   166,     5,
     170,    -1,    89,     1,    -1,    90,   166,     5,   166,     5,
     166,    -1,    90,   166,     5,   170,     5,   166,    -1,    90,
       1,    -1,    91,   166,     5,   166,     5,   164,    -1,    91,
     166,     5,   170,     5,   164,    -1,    91,     1,    -1,    92,
     166,     5,   166,     5,   166,    -1,    92,   166,     5,   170,
       5,   166,    -1,    92,     1,    -1,    93,   166,     5,   166,
      -1,    93,   166,     5,   170,    -1,    93,     1,    -1,    94,
      44,     5,   164,     5,   164,    -1,    94,     4,     5,   164,
       5,   164,    -1,    94,     1,    -1,    95,     4,     5,     4,
       5,    44,    -1,    95,    44,     5,    44,     5,    44,    -1,
      95,     1,    -1,    96,    44,    -1,    96,     4,    -1,    96,
       1,    -1,    97,    44,    -1,    97,     1,    -1,    99,    44,
      -1,    99,     1,    -1,   100,    44,    -1,   100,     1,    -1,
     101,   165,     5,   165,     5,   168,    -1,   101,     1,    -1,
     102,   165,    -1,   102,     1,    -1,   123,   164,    -1,   123,
       1,    -1,   103,     4,    -1,   103,    44,    -1,   103,     1,
      -1,   106,   164,    -1,   106,     1,    -1,   104,     4,     5,
     164,    -1,   104,    44,     5,   164,    -1,   104,     1,    -1,
     105,     4,     5,   164,    -1,   105,    44,     5,   164,    -1,
     105,     1,    -1,    79,    44,     5,     4,    -1,    79,    44,
       5,    44,    -1,    79,     1,    -1,   121,     4,    -1,   121,
      44,    -1,   121,     1,    -1,    76,    -1,    76,     1,    -1,
      78,    -1,    78,     1,    -1,    77,   164,    -1,    77,     1,
      -1,   119,    -1,   119,     1,    -1,   122,    44,    -1,   122,
       1,    -1,   144,    -1,   144,     1,    -1,   117,    44,     5,
     166,     5,   247,    -1,   117,     1,    -1,   118,    44,     5,
     166,     5,   247,    -1,   118,     1,    -1,    44,     5,     4,
      -1,   247,     5,    44,     5,     4,    -1,   125,    44,     5,
     164,    -1,   125,     4,     5,   164,    -1,   125,     1,    -1,
     126,    41,     5,    41,    -1,   126,    41,     5,    44,    -1,
     126,    44,     5,    41,    -1,   126,    44,     5,    44,    -1,
     126,     1,    -1,   127,    41,     5,    41,    -1,   127,    41,
       5,    44,    -1,   127,    44,     5,    41,    -1,   127,    44,
       5,    44,    -1,   127,     1,    -1,   128,    41,     5,    41,
      -1,   128,    41,     5,    44,    -1,   128,    44,     5,    41,
      -1,   128,    44,     5,    44,    -1,   128,     1,    -1,   129,
      41,    -1,   129,    44,    -1,   129,     1,    -1,   131,   164,
       5,   164,    -1,   131,     1,    -1,   132,   164,     5,   164,
      -1,   132,     1,    -1,   133,   166,     5,   165,    -1,   133,
       1,    -1,   134,   166,     5,   165,    -1,   134,     1,    -1,
     135,   166,     5,   165,    -1,   135,     1,    -1,   130,   166,
       5,   165,    -1,   130,     1,    -1,    69,   166,     5,   165,
      -1,    69,     1,    -1,   136,   166,    -1,   136,     1,    -1,
     137,   166,     5,   166,    -1,   137,   166,     5,    44,    -1,
     137,     1,    -1,   138,   166,     5,   166,    -1,   138,   166,
       5,    44,    -1,   138,     1,    -1,   139,   166,     5,   166,
      -1,   139,   166,     5,    44,    -1,   139,     1,    -1,   140,
     166,     5,   166,    -1,   140,   166,     5,    44,    -1,   140,
       1,    -1,   141,   164,     5,   165,    -1,   141,     1,    -1,
     142,   164,     5,   165,    -1,   142,     1,    -1,   143,   164,
       5,   165,    -1,   143,     1,    -1,   146,   166,    -1,   146,
       1,    -1,   147,   166,    -1,   147,     1,    -1,   148,   164,
       5,   164,    -1,   148,     1,    -1,   149,   164,     5,   164,
      -1,   149,     1,    -1,   150,   164,     5,   164,    -1,   150,
       1,    -1,   151,   164,     5,   164,    -1,   151,     1,    -1,
     152,   164,     5,   164,    -1,   152,     1,    -1,   153,   164,
       5,   164,    -1,   153,     1,    -1,   154,   164,     5,   164,
      -1,   154,     1,    -1,   155,   171,    -1,   155,   166,    -1,
     155,     1,    -1,   156,   171,    -1,   156,   166,    -1,   156,
       1,    -1,   157,   164,    -1,   157,     1,    -1,   158,   164,
      -1,   158,     1,    -1,   159,   166,     5,   164,    -1,   159,
       1,    -1,   160,   171,     5,   164,    -1,   160,     1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   231,   231,   233,   237,   238,   239,   240,   241,   242,
     245,   245,   246,   246,   247,   247,   248,   248,   248,   248,
     248,   248,   250,   250,   251,   251,   251,   251,   252,   252,
     252,   253,   253,   254,   254,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   283,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   311,   312,   313,   314,   319,   325,
     334,   335,   339,   340,   343,   347,   348,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414,
     415,   416,   417,   418,   419,   420,   421,   422,   423,   424,
     425,   426,   427,   428,   429,   430,   431,   432,   433,   434,
     435,   436,   437,   438,   439,   440,   441,   442,   443,   444,
     445,   446,   447,   448,   449,   450,   451,   452,   453,   454,
     458,   459,   463,   464,   468,   469,   473,   474,   478,   479,
     484,   485,   489,   490,   494,   495,   499,   500,   505,   506,
     510,   511,   515,   516,   520,   521,   526,   527,   531,   532,
     536,   537,   541,   542,   546,   547,   551,   552,   556,   557,
     558,   562,   563,   567,   568,   573,   574,   578,   579,   584,
     585,   589,   590,   594,   595,   596,   600,   601,   602,   606,
     607,   611,   612,   617,   618,   619,   623,   624,   629,   630,
     634,   635,   639,   640,   645,   646,   647,   651,   652,   653,
     657,   658,   659,   663,   664,   665,   669,   670,   671,   675,
     676,   677,   681,   682,   683,   687,   688,   689,   693,   694,
     695,   699,   700,   701,   705,   706,   707,   711,   712,   713,
     717,   718,   719,   723,   724,   728,   729,   733,   734,   738,
     739,   743,   744,   748,   749,   753,   754,   755,   759,   760,
     764,   765,   766,   770,   771,   772,   777,   778,   779,   783,
     784,   785,   789,   790,   794,   795,   799,   800,   804,   805,
     809,   810,   814,   815,   819,   820,   824,   825,   829,   838,
     847,   848,   849,   853,   854,   855,   856,   857,   861,   862,
     863,   864,   865,   869,   870,   871,   872,   873,   877,   878,
     879,   883,   884,   888,   889,   893,   894,   898,   899,   903,
     904,   908,   909,   913,   914,   918,   919,   923,   924,   925,
     929,   930,   931,   935,   936,   937,   941,   942,   943,   948,
     949,   953,   954,   958,   959,   963,   964,   968,   969,   973,
     974,   978,   979,   983,   984,   988,   989,   993,   994,   998,
     999,  1003,  1004,  1008,  1009,  1010,  1014,  1015,  1016,  1020,
    1021,  1025,  1026,  1031,  1032,  1036,  1037
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
  "DMODULE", "DLOAD", "DIMPORT", "DSWITCH", "DSELECT", "DCASE",
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
  "I_TRAC", "I_WRT", "I_STO", "I_FORB", "$accept", "input", "line",
  "xoperand", "operand", "op_variable", "op_register", "x_op_immediate",
  "op_immediate", "op_scalar", "op_string", "string_or_name", "directive",
  "has_symlist", "hasnt_symlist", "label", "def_line", "instruction",
  "inst_ld", "inst_ldrf", "inst_ldnil", "inst_add", "inst_adds",
  "inst_sub", "inst_subs", "inst_mul", "inst_muls", "inst_div",
  "inst_divs", "inst_mod", "inst_pow", "inst_eq", "inst_ne", "inst_ge",
  "inst_gt", "inst_le", "inst_lt", "inst_try", "inst_inc", "inst_dec",
  "inst_incp", "inst_decp", "inst_neg", "inst_not", "inst_call",
  "inst_inst", "inst_unpk", "inst_unps", "inst_push", "inst_pshr",
  "inst_pop", "inst_peek", "inst_xpop", "inst_ldv", "inst_ldvt",
  "inst_stv", "inst_stvr", "inst_stvs", "inst_ldp", "inst_ldpt",
  "inst_stp", "inst_stpr", "inst_stps", "inst_trav", "inst_tran",
  "inst_tral", "inst_ipop", "inst_gena", "inst_gend", "inst_genr",
  "inst_geor", "inst_ris", "inst_jmp", "inst_bool", "inst_ift", "inst_iff",
  "inst_fork", "inst_jtry", "inst_ret", "inst_reta", "inst_retval",
  "inst_nop", "inst_ptry", "inst_end", "inst_swch", "inst_sele",
  "switch_list", "inst_once", "inst_band", "inst_bor", "inst_bxor",
  "inst_bnot", "inst_and", "inst_or", "inst_ands", "inst_ors", "inst_xors",
  "inst_mods", "inst_pows", "inst_nots", "inst_has", "inst_hasn",
  "inst_give", "inst_givn", "inst_in", "inst_noin", "inst_prov",
  "inst_psin", "inst_pass", "inst_shr", "inst_shl", "inst_shrs",
  "inst_shls", "inst_ldvr", "inst_ldpr", "inst_lsb", "inst_indi",
  "inst_stex", "inst_trac", "inst_wrt", "inst_sto", "inst_forb", 0
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
     415
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   161,   162,   162,   163,   163,   163,   163,   163,   163,
     164,   164,   165,   165,   166,   166,   167,   167,   167,   167,
     167,   167,   168,   168,   169,   169,   169,   169,   170,   170,
     170,   171,   171,   172,   172,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     174,   174,   175,   175,   176,   177,   177,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     179,   179,   180,   180,   181,   181,   182,   182,   183,   183,
     184,   184,   185,   185,   186,   186,   187,   187,   188,   188,
     189,   189,   190,   190,   191,   191,   192,   192,   193,   193,
     194,   194,   195,   195,   196,   196,   197,   197,   198,   198,
     198,   199,   199,   200,   200,   201,   201,   202,   202,   203,
     203,   204,   204,   205,   205,   205,   206,   206,   206,   207,
     207,   208,   208,   209,   209,   209,   210,   210,   211,   211,
     212,   212,   213,   213,   214,   214,   214,   215,   215,   215,
     216,   216,   216,   217,   217,   217,   218,   218,   218,   219,
     219,   219,   220,   220,   220,   221,   221,   221,   222,   222,
     222,   223,   223,   223,   224,   224,   224,   225,   225,   225,
     226,   226,   226,   227,   227,   228,   228,   229,   229,   230,
     230,   231,   231,   232,   232,   233,   233,   233,   234,   234,
     235,   235,   235,   236,   236,   236,   237,   237,   237,   238,
     238,   238,   239,   239,   240,   240,   241,   241,   242,   242,
     243,   243,   244,   244,   245,   245,   246,   246,   247,   247,
     248,   248,   248,   249,   249,   249,   249,   249,   250,   250,
     250,   250,   250,   251,   251,   251,   251,   251,   252,   252,
     252,   253,   253,   254,   254,   255,   255,   256,   256,   257,
     257,   258,   258,   259,   259,   260,   260,   261,   261,   261,
     262,   262,   262,   263,   263,   263,   264,   264,   264,   265,
     265,   266,   266,   267,   267,   268,   268,   269,   269,   270,
     270,   271,   271,   272,   272,   273,   273,   274,   274,   275,
     275,   276,   276,   277,   277,   277,   278,   278,   278,   279,
     279,   280,   280,   281,   281,   282,   282
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     1,     2,     3,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     3,     4,     4,
       5,     3,     4,     3,     4,     3,     3,     2,     3,     3,
       4,     1,     2,     2,     3,     4,     4,     5,     5,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     6,     1,
       3,     3,     3,     2,     2,     4,     5,     3,     4,     2,
       3,     2,     1,     2,     2,     3,     2,     2,     2,     2,
       1,     3,     1,     3,     2,     0,     1,     1,     1,     1,
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
       4,     2,     4,     2,     2,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     4,     4,     2,     4,     4,     2,     4,
       2,     4,     2,     2,     1,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     4,     4,     2,     6,     6,     2,
       6,     6,     2,     6,     6,     2,     4,     4,     2,     4,
       4,     2,     6,     6,     2,     6,     6,     2,     6,     6,
       2,     4,     4,     2,     6,     6,     2,     6,     6,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     6,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       4,     4,     2,     4,     4,     2,     4,     4,     2,     2,
       2,     2,     1,     2,     1,     2,     2,     2,     1,     2,
       2,     2,     1,     2,     6,     2,     6,     2,     3,     5,
       4,     4,     2,     4,     4,     4,     4,     2,     4,     4,
       4,     4,     2,     4,     4,     4,     4,     2,     2,     2,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     2,     2,     4,     4,     2,
       4,     4,     2,     4,     4,     2,     4,     4,     2,     4,
       2,     4,     2,     4,     2,     2,     2,     2,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     4,     2,     4,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,     0,     4,    35,     0,     0,     0,     0,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
      82,     0,     0,     0,     0,     0,     0,     0,     0,    69,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   264,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     3,     0,     0,     0,    97,   162,    98,
      99,   105,   100,   106,   101,   107,   102,   108,   103,   104,
     113,   114,   115,   116,   117,   118,   156,   109,   110,   111,
     112,   119,   120,   150,   151,   148,   149,   121,   122,   123,
     184,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     134,   133,   135,   136,   137,   138,   139,   140,   141,   142,
     159,   143,   144,   145,   146,   147,   158,   152,   154,   153,
     155,   157,   163,   160,   161,   183,   164,   165,   166,   167,
     168,   169,   170,   171,   172,   174,   175,   173,   176,   177,
     178,   179,   180,   181,   182,   185,   186,   187,   188,   189,
     190,   191,   192,   193,   194,   195,   196,   197,   198,   199,
       9,    95,     0,     0,    95,    95,    95,    47,    95,     0,
       0,    95,    79,    81,    84,    95,    36,    52,    53,    95,
      14,    16,    17,    18,    19,    20,    21,     0,    15,     0,
       0,     0,     0,     0,     0,    86,    87,    88,    89,    90,
      73,    92,    74,    83,     0,    94,   201,     0,   205,   204,
     207,    30,    24,    29,    28,    25,    26,    10,     0,    11,
      12,    13,    27,   211,     0,   215,     0,   219,     0,   223,
       0,   225,     0,   209,     0,   213,     0,   217,     0,   221,
       0,   394,     0,   242,   241,   244,   243,   246,   245,   248,
     247,   250,   249,   252,   251,   343,   347,   346,   345,   338,
       0,   265,   263,   267,   266,   269,   268,   276,     0,   279,
       0,   282,     0,   285,     0,   288,     0,   291,     0,   294,
       0,   297,     0,   300,     0,   303,     0,   306,     0,     0,
     309,     0,     0,   312,   311,   310,   314,   313,   273,   272,
     316,   315,   318,   317,   320,     0,   322,   321,   327,   325,
     326,   332,     0,     0,   335,     0,     0,   329,   328,   227,
       0,   229,     0,   233,     0,   231,     0,   237,     0,   235,
       0,   260,     0,   262,     0,   255,     0,   258,     0,   355,
       0,   357,     0,   349,   240,   238,   239,   341,   339,   340,
     351,   350,   324,   323,   203,     0,   362,     0,     0,   367,
       0,     0,   372,     0,     0,   377,     0,     0,   380,   378,
     379,   392,     0,   382,     0,   384,     0,   386,     0,   388,
       0,   390,     0,   396,   395,   399,     0,   402,     0,   405,
       0,   408,     0,   410,     0,   412,     0,   414,     0,   353,
     271,   270,   416,   415,   418,   417,   420,     0,   422,     0,
     424,     0,   426,     0,   428,     0,   430,     0,   432,     0,
     435,    32,    31,   434,   433,   438,   437,   436,   440,   439,
     442,   441,   444,     0,   446,     0,     5,     7,     0,     8,
      96,    37,    22,    95,    23,    41,    43,    45,    46,    48,
      49,    71,    70,    72,    77,    80,    85,    54,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    95,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     6,    38,    39,    42,    44,    50,
      78,    55,    56,    59,    60,    61,    62,    67,    64,     0,
      65,    66,    63,    91,    93,    75,   200,   206,   210,   214,
     218,   222,   224,   208,   212,   216,   220,   393,   336,   337,
     274,   275,     0,     0,     0,     0,     0,     0,   286,   287,
     289,   290,     0,     0,     0,     0,     0,     0,   301,   302,
       0,     0,     0,     0,     0,   330,   331,   333,   334,   226,
     228,   232,   230,   236,   234,   259,   261,   254,   253,   257,
     256,     0,     0,   202,   361,   360,   363,   364,   365,   366,
     368,   369,   370,   371,   373,   374,   375,   376,   391,   381,
     383,   385,   387,   389,   398,   397,   401,   400,   404,   403,
     407,   406,   409,   411,   413,   419,   421,   423,   425,   427,
     429,   431,   443,   445,    40,    34,    33,    57,    58,     0,
      76,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      68,   277,   278,   280,   281,   283,   284,   292,   293,   295,
     296,   298,   299,   305,   304,   307,   308,   319,     0,   354,
     356,     0,     0,   358,     0,     0,   359
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,   143,   308,   309,   310,   278,   533,   311,   312,
     514,   757,   144,   290,   292,   145,   531,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   202,   203,   204,   205,   206,   207,   208,
     209,   210,   211,   212,   213,   214,   799,   215,   216,   217,
     218,   219,   220,   221,   222,   223,   224,   225,   226,   227,
     228,   229,   230,   231,   232,   233,   234,   235,   236,   237,
     238,   239,   240,   241,   242,   243,   244,   245,   246,   247,
     248,   249
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -249
static const yytype_int16 yypact[] =
{
    -249,   313,  -249,    16,  -249,  -249,    34,    93,    96,   167,
     170,   179,   185,  -249,   186,   215,   233,   236,   237,   216,
    -249,   266,   267,   271,     5,   284,   986,   986,   434,  -249,
     248,   241,   243,   244,   268,   269,   278,   318,   306,    97,
     316,    82,   159,   177,   198,   214,   232,   632,   729,   770,
    1333,  1348,  1362,  1372,  1383,  1398,   253,   555,   119,   658,
     169,    17,   673,   714,  -249,  1413,  1424,  1434,  1448,  1463,
    1475,  1485,  1498,  1513,  1526,  1536,     6,   176,   314,    55,
    1548,    66,    70,   473,   536,   489,   494,   495,   742,   941,
     956,   971,   997,  1012,  1027,  1563,   172,   501,   514,   515,
     567,   235,   496,   500,   568,  1053,  1577,   616,    72,   138,
     140,   147,  1587,  1068,  1083,  1598,  1613,  1628,  1639,  1649,
    1663,  1678,  1690,  1109,  1124,  1139,   288,  1700,  1713,  1728,
    1165,  1180,  1195,  1221,  1236,  1251,  1277,    29,   116,  1292,
    1307,  1741,     7,  -249,   351,   781,   357,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,   323,   754,   754,   323,   323,   323,   326,   323,   696,
     330,   323,   446,  -249,  -249,   323,  -249,  -249,  -249,   323,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,   472,  -249,   486,
     508,   153,   527,   562,   565,  -249,  -249,  -249,  -249,  -249,
     573,  -249,   574,  -249,   490,  -249,  -249,   609,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,   610,  -249,
    -249,  -249,  -249,  -249,   611,  -249,   613,  -249,   614,  -249,
     627,  -249,   630,  -249,   631,  -249,   641,  -249,   642,  -249,
     643,  -249,   656,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
     657,  -249,  -249,  -249,  -249,  -249,  -249,  -249,   659,  -249,
     663,  -249,   664,  -249,   665,  -249,   666,  -249,   667,  -249,
     678,  -249,   679,  -249,   680,  -249,   681,  -249,   691,   692,
    -249,   693,   695,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,   724,  -249,  -249,  -249,  -249,
    -249,  -249,   727,   728,  -249,   733,   734,  -249,  -249,  -249,
     739,  -249,   747,  -249,   748,  -249,   751,  -249,   752,  -249,
     768,  -249,   775,  -249,   776,  -249,   777,  -249,   780,  -249,
     794,  -249,   796,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,   797,  -249,   798,   799,  -249,
     805,   808,  -249,   809,   816,  -249,   817,   818,  -249,  -249,
    -249,  -249,   819,  -249,   824,  -249,   825,  -249,   828,  -249,
     829,  -249,   834,  -249,  -249,  -249,   939,  -249,   940,  -249,
     948,  -249,   953,  -249,   976,  -249,   978,  -249,   979,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  1025,  -249,  1032,
    -249,  1034,  -249,  1035,  -249,  1078,  -249,  1081,  -249,  1082,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  1084,  -249,  1085,  -249,  -249,   595,  -249,
    -249,   621,  -249,   323,  -249,  1046,  1049,  -249,  -249,  -249,
    1050,  -249,  -249,  -249,  1051,  -249,  -249,    11,    28,   105,
     633,   697,   690,  1091,  1092,  1135,  1101,  1102,   323,   915,
     915,   915,   915,   915,   915,   915,   915,   915,   915,   915,
    1761,   230,   902,   902,   902,   902,   902,   902,   902,   902,
     902,   902,   915,   915,  1140,  1103,  1761,   915,   915,   915,
     915,   915,   915,   915,   915,   915,   915,   986,   986,   593,
    1751,   986,   986,   915,   915,   915,   -24,    -8,    -7,    24,
      43,    44,  1761,   915,   915,  1761,  1761,  1761,   435,   462,
     516,   930,  1761,  1761,  1761,   915,   915,   915,   915,   915,
     915,   915,   915,   915,  -249,  -249,  1104,  -249,  -249,  -249,
    -249,    12,    12,  -249,  -249,  -249,  -249,  -249,  -249,  1143,
    -249,  -249,  -249,  -249,  -249,  1107,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  1146,  1147,  1190,  1193,  1194,  1196,  -249,  -249,
    -249,  -249,  1197,  1198,  1199,  1200,  1202,  1203,  -249,  -249,
    1246,  1249,  1250,  1252,  1253,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  1254,  1255,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  1141,
    -249,   986,   986,   915,   915,   915,   915,   986,   986,   915,
     915,   986,   986,   915,   915,  1156,  1212,   754,  1217,  1217,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  1258,  1259,
    1259,  1303,  1266,  -249,  1306,  1308,  -249
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -249,  -249,  -249,    62,   -81,   -26,  -249,  -247,  -248,  1245,
    -107,   671,  -249,  -249,  -249,  -249,  -203,  1169,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,   537,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -353
static const yytype_int16 yytable[] =
{
     277,   279,   395,   397,   534,   534,   535,   377,   524,   267,
     378,   534,   542,   297,   299,   641,   755,   716,   349,   250,
     717,   324,   326,   328,   330,   332,   334,   336,   338,   340,
     510,   517,   643,   718,   720,   525,   719,   721,   251,   356,
     358,   360,   362,   364,   366,   368,   370,   372,   374,   376,
     379,   536,   537,   538,   389,   540,   386,   268,   544,   511,
     512,   350,   546,   642,   756,   722,   547,   390,   723,   422,
     270,   392,   644,   449,   271,   272,   273,   274,   275,   276,
     445,   511,   512,   300,   724,   726,   462,   725,   727,   468,
     470,   472,   474,   476,   478,   480,   482,   252,   296,   387,
     253,   491,   493,   495,   314,   316,   318,   320,   322,   645,
     391,   513,   516,   450,   393,   523,   451,   515,   342,   344,
     345,   347,  -342,   270,   352,   354,   301,   271,   272,   273,
     274,   275,   276,   302,   303,   304,   305,   306,   270,   452,
     307,   455,   271,   272,   273,   274,   275,   276,   458,   646,
     408,   410,   412,   414,   416,   418,   420,   270,   551,   552,
     313,   271,   272,   273,   274,   275,   276,   443,   511,   512,
     348,   254,  -344,   423,   255,   464,   466,   380,   315,   453,
     381,   456,   454,   256,   457,   484,   486,   488,   459,   257,
     258,   460,   497,   499,   501,   503,   505,   507,   509,   317,
     270,   519,   521,   301,   271,   272,   273,   274,   275,   276,
     302,   303,   304,   305,   306,   319,   424,   307,   270,   259,
     382,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   321,   668,   307,   433,   260,  -348,   270,
     261,   262,   301,   271,   272,   273,   274,   275,   276,   302,
     303,   304,   305,   306,   341,   270,   307,   263,   301,   271,
     272,   273,   274,   275,   276,   302,   303,   304,   305,   306,
     264,   265,   307,   270,   669,   266,   301,   271,   272,   273,
     274,   275,   276,   302,   303,   304,   305,   306,   269,   489,
     307,  -352,   285,   286,   270,   287,   288,   301,   271,   272,
     273,   274,   275,   276,   302,   303,   304,   305,   306,   289,
     291,   307,   295,     2,     3,   383,     4,   298,   384,   293,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
     636,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,   526,   655,    38,   270,   385,   294,
     529,   271,   272,   273,   274,   275,   276,   530,   539,    39,
      40,   543,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   394,   280,   270,   548,   281,   734,
     271,   272,   273,   274,   275,   276,   282,   283,   545,   667,
     398,   549,   284,   399,   558,   401,   404,   434,   402,   405,
     435,   437,   425,   270,   438,   694,   736,   271,   272,   273,
     274,   275,   276,   550,   270,   427,   429,   301,   271,   272,
     273,   274,   275,   276,   302,   303,   304,   305,   306,   534,
     797,   728,   553,   400,   731,   732,   733,   396,   403,   406,
     436,   742,   743,   744,   439,   426,   670,   672,   674,   676,
     678,   680,   682,   684,   686,   688,   343,   270,   428,   430,
     738,   271,   272,   273,   274,   275,   276,   554,   431,   440,
     555,   705,   706,   708,   710,   711,   712,   270,   556,   557,
     301,   271,   272,   273,   274,   275,   276,   302,   303,   304,
     305,   306,   735,   737,   739,   741,   270,   707,   634,   301,
     271,   272,   273,   274,   275,   276,   302,   303,   304,   305,
     306,   432,   441,   307,   559,   560,   561,   446,   562,   563,
     447,   656,   657,   658,   659,   660,   661,   662,   663,   664,
     665,   666,   564,   323,   270,   565,   566,   647,   271,   272,
     273,   274,   275,   276,   690,   691,   567,   568,   569,   695,
     696,   697,   698,   699,   700,   701,   702,   703,   704,   346,
     448,   570,   571,   635,   572,   713,   714,   715,   573,   574,
     575,   576,   577,   270,   351,   729,   730,   271,   272,   273,
     274,   275,   276,   578,   579,   580,   581,   745,   746,   747,
     748,   749,   750,   751,   752,   753,   582,   583,   584,   270,
     585,   648,   301,   271,   272,   273,   274,   275,   276,   302,
     303,   304,   305,   306,   270,   353,   307,   301,   271,   272,
     273,   274,   275,   276,   302,   303,   304,   305,   306,   586,
     325,   307,   587,   588,   649,   781,   782,   541,   589,   590,
     301,   787,   788,   407,   591,   791,   792,   302,   303,   304,
     305,   306,   592,   593,   532,   270,   594,   595,   301,   271,
     272,   273,   274,   275,   276,   302,   303,   304,   305,   306,
     270,   327,   307,   596,   271,   272,   273,   274,   275,   276,
     597,   598,   599,   270,   527,   600,   301,   271,   272,   273,
     274,   275,   276,   302,   303,   304,   305,   306,   301,   601,
     307,   602,   603,   604,   605,   302,   303,   304,   305,   306,
     606,   270,   532,   607,   608,   271,   272,   273,   274,   275,
     276,   609,   610,   611,   612,   783,   784,   785,   786,   613,
     614,   789,   790,   615,   616,   793,   794,    39,    40,   617,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   142,   409,   270,   618,   619,   301,   271,   272,   273,
     274,   275,   276,   620,   303,   304,   270,   411,   621,   301,
     271,   272,   273,   274,   275,   276,   302,   303,   304,   305,
     306,   270,   413,   307,   740,   271,   272,   273,   274,   275,
     276,   622,   270,   623,   624,   301,   271,   272,   273,   274,
     275,   276,   302,   303,   304,   305,   306,   270,   415,   307,
     301,   271,   272,   273,   274,   275,   276,   302,   303,   304,
     305,   306,   270,   417,   307,   301,   271,   272,   273,   274,
     275,   276,   302,   303,   304,   305,   306,   270,   419,   307,
     625,   271,   272,   273,   274,   275,   276,   626,   270,   627,
     628,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   270,   442,   307,   301,   271,   272,   273,
     274,   275,   276,   302,   303,   304,   305,   306,   270,   463,
     307,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   629,   465,   307,   630,   631,   637,   632,
     633,   638,   639,   640,   270,   650,   651,   301,   271,   272,
     273,   274,   275,   276,   302,   303,   304,   305,   306,   270,
     483,   307,   301,   271,   272,   273,   274,   275,   276,   302,
     303,   304,   305,   306,   270,   485,   307,   301,   271,   272,
     273,   274,   275,   276,   302,   303,   304,   305,   306,   652,
     487,   307,   653,   654,   692,   780,   754,   693,   759,   760,
     270,   761,   762,   301,   271,   272,   273,   274,   275,   276,
     302,   303,   304,   305,   306,   270,   496,   307,   301,   271,
     272,   273,   274,   275,   276,   302,   303,   304,   305,   306,
     270,   498,   307,   301,   271,   272,   273,   274,   275,   276,
     302,   303,   304,   305,   306,   763,   500,   307,   764,   765,
     795,   766,   767,   768,   769,   770,   270,   771,   772,   301,
     271,   272,   273,   274,   275,   276,   302,   303,   304,   305,
     306,   270,   502,   307,   301,   271,   272,   273,   274,   275,
     276,   302,   303,   304,   305,   306,   270,   504,   307,   301,
     271,   272,   273,   274,   275,   276,   302,   303,   304,   305,
     306,   773,   506,   307,   774,   775,   796,   776,   777,   778,
     779,   798,   270,   801,   802,   301,   271,   272,   273,   274,
     275,   276,   302,   303,   304,   305,   306,   270,   508,   307,
     301,   271,   272,   273,   274,   275,   276,   302,   303,   304,
     305,   306,   270,   518,   307,   301,   271,   272,   273,   274,
     275,   276,   302,   303,   304,   305,   306,   803,   520,   307,
     804,   805,   806,   758,   528,     0,   800,     0,   270,     0,
       0,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   270,   329,   307,   301,   271,   272,   273,
     274,   275,   276,   302,   303,   304,   305,   306,   270,   331,
     307,   301,   271,   272,   273,   274,   275,   276,   302,   303,
     304,   305,   306,   333,     0,   307,     0,     0,     0,     0,
       0,     0,     0,   335,   270,     0,     0,     0,   271,   272,
     273,   274,   275,   276,   337,     0,     0,     0,     0,   270,
       0,     0,     0,   271,   272,   273,   274,   275,   276,   339,
       0,     0,     0,   270,     0,     0,     0,   271,   272,   273,
     274,   275,   276,   270,   355,     0,     0,   271,   272,   273,
     274,   275,   276,     0,   270,   357,     0,     0,   271,   272,
     273,   274,   275,   276,     0,   359,     0,     0,     0,   270,
       0,     0,     0,   271,   272,   273,   274,   275,   276,   361,
       0,     0,     0,     0,   270,     0,     0,     0,   271,   272,
     273,   274,   275,   276,   363,   270,     0,     0,     0,   271,
     272,   273,   274,   275,   276,   270,   365,     0,     0,   271,
     272,   273,   274,   275,   276,     0,   367,     0,     0,   270,
       0,     0,     0,   271,   272,   273,   274,   275,   276,   369,
       0,     0,     0,     0,   270,     0,     0,     0,   271,   272,
     273,   274,   275,   276,   371,     0,   270,     0,     0,     0,
     271,   272,   273,   274,   275,   276,   270,   373,     0,     0,
     271,   272,   273,   274,   275,   276,     0,   375,     0,   270,
       0,     0,     0,   271,   272,   273,   274,   275,   276,   388,
       0,     0,     0,     0,   270,     0,     0,     0,   271,   272,
     273,   274,   275,   276,   421,     0,     0,   270,     0,     0,
       0,   271,   272,   273,   274,   275,   276,   270,   444,     0,
       0,   271,   272,   273,   274,   275,   276,     0,   461,   270,
       0,     0,     0,   271,   272,   273,   274,   275,   276,   467,
       0,     0,     0,     0,   270,     0,     0,     0,   271,   272,
     273,   274,   275,   276,   469,     0,     0,     0,   270,     0,
       0,     0,   271,   272,   273,   274,   275,   276,   270,   471,
       0,     0,   271,   272,   273,   274,   275,   276,     0,   270,
     473,     0,     0,   271,   272,   273,   274,   275,   276,     0,
     475,     0,     0,     0,   270,     0,     0,     0,   271,   272,
     273,   274,   275,   276,   477,     0,     0,     0,     0,   270,
       0,     0,     0,   271,   272,   273,   274,   275,   276,   479,
     270,     0,     0,     0,   271,   272,   273,   274,   275,   276,
     270,   481,     0,     0,   271,   272,   273,   274,   275,   276,
       0,   490,     0,     0,   270,     0,     0,     0,   271,   272,
     273,   274,   275,   276,   492,     0,     0,     0,     0,   270,
       0,     0,     0,   271,   272,   273,   274,   275,   276,   494,
       0,   270,     0,     0,     0,   271,   272,   273,   274,   275,
     276,   270,   522,     0,     0,   271,   272,   273,   274,   275,
     276,     0,     0,     0,   270,   709,     0,     0,   271,   272,
     273,   274,   275,   276,     0,     0,     0,     0,     0,   270,
       0,     0,     0,   271,   272,   273,   274,   275,   276,     0,
       0,     0,   270,     0,     0,     0,   271,   272,   273,   274,
     275,   276,   270,     0,     0,     0,   271,   272,   273,   274,
     275,   276,   270,     0,     0,   301,   271,   272,   273,   274,
     275,   276,   302,   303,   304,   305,   306,   671,   673,   675,
     677,   679,   681,   683,   685,   687,   689
};

static const yytype_int16 yycheck[] =
{
      26,    27,    83,    84,   252,   253,   253,     1,     1,     4,
       4,   259,   259,    39,    40,     4,     4,    41,     1,     3,
      44,    47,    48,    49,    50,    51,    52,    53,    54,    55,
       1,   138,     4,    41,    41,   142,    44,    44,     4,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      44,   254,   255,   256,    80,   258,     1,    52,   261,    52,
      53,    44,   265,    52,    52,    41,   269,     1,    44,    95,
      41,     1,    44,     1,    45,    46,    47,    48,    49,    50,
     106,    52,    53,     1,    41,    41,   112,    44,    44,   115,
     116,   117,   118,   119,   120,   121,   122,     4,     1,    44,
       4,   127,   128,   129,    42,    43,    44,    45,    46,     4,
      44,   137,   138,    41,    44,   141,    44,     1,    56,    57,
       1,    59,     3,    41,    62,    63,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    41,     1,
      58,     1,    45,    46,    47,    48,    49,    50,     1,    44,
      88,    89,    90,    91,    92,    93,    94,    41,     5,     6,
       1,    45,    46,    47,    48,    49,    50,   105,    52,    53,
       1,     4,     3,     1,     4,   113,   114,     1,     1,    41,
       4,    41,    44,     4,    44,   123,   124,   125,    41,     4,
       4,    44,   130,   131,   132,   133,   134,   135,   136,     1,
      41,   139,   140,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,     1,    44,    58,    41,     4,
      44,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,     1,     4,    58,     1,     4,     3,    41,
       4,     4,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,     1,    41,    58,    41,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
       4,     4,    58,    41,    44,     4,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,     4,     1,
      58,     3,    44,    52,    41,    52,    52,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    41,
      41,    58,     6,     0,     1,     1,     3,     1,     4,    41,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
     533,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,     3,   558,    43,    41,    44,    41,
       3,    45,    46,    47,    48,    49,    50,    44,    42,    56,
      57,    41,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,     1,    41,    41,     5,    44,    44,
      45,    46,    47,    48,    49,    50,    52,    53,    42,   570,
       1,     5,    58,     4,     4,     1,     1,     1,     4,     4,
       4,     1,     1,    41,     4,   586,    44,    45,    46,    47,
      48,    49,    50,     5,    41,     1,     1,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,   777,
     777,   612,     5,    44,   615,   616,   617,     1,    44,    44,
      44,   622,   623,   624,    44,    44,   572,   573,   574,   575,
     576,   577,   578,   579,   580,   581,     1,    41,    44,    44,
      44,    45,    46,    47,    48,    49,    50,     5,     1,     1,
       5,   597,   598,   599,   600,   601,   602,    41,     5,     5,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,   618,   619,   620,   621,    41,     4,     3,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    44,    44,    58,     5,     5,     5,     1,     5,     5,
       4,   559,   560,   561,   562,   563,   564,   565,   566,   567,
     568,   569,     5,     1,    41,     5,     5,     4,    45,    46,
      47,    48,    49,    50,   582,   583,     5,     5,     5,   587,
     588,   589,   590,   591,   592,   593,   594,   595,   596,     1,
      44,     5,     5,    42,     5,   603,   604,   605,     5,     5,
       5,     5,     5,    41,     1,   613,   614,    45,    46,    47,
      48,    49,    50,     5,     5,     5,     5,   625,   626,   627,
     628,   629,   630,   631,   632,   633,     5,     5,     5,    41,
       5,     4,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    41,     1,    58,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,     5,
       1,    58,     5,     5,    44,   761,   762,    41,     5,     5,
      44,   767,   768,     1,     5,   771,   772,    51,    52,    53,
      54,    55,     5,     5,    58,    41,     5,     5,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      41,     1,    58,     5,    45,    46,    47,    48,    49,    50,
       5,     5,     5,    41,     3,     5,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    44,     5,
      58,     5,     5,     5,     5,    51,    52,    53,    54,    55,
       5,    41,    58,     5,     5,    45,    46,    47,    48,    49,
      50,     5,     5,     5,     5,   763,   764,   765,   766,     5,
       5,   769,   770,     5,     5,   773,   774,    56,    57,     5,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,     1,    41,     5,     5,    44,    45,    46,    47,
      48,    49,    50,     5,    52,    53,    41,     1,     5,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    41,     1,    58,    44,    45,    46,    47,    48,    49,
      50,     5,    41,     5,     5,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    41,     1,    58,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    41,     1,    58,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    41,     1,    58,
       5,    45,    46,    47,    48,    49,    50,     5,    41,     5,
       5,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    41,     1,    58,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    41,     1,
      58,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,     5,     1,    58,     5,     5,    42,     5,
       5,    42,    42,    42,    41,     4,     4,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    41,
       1,    58,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    41,     1,    58,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,     4,
       1,    58,    41,    41,     4,     4,    42,    44,     5,    42,
      41,     5,     5,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    41,     1,    58,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      41,     1,    58,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,     5,     1,    58,     5,     5,
      44,     5,     5,     5,     5,     5,    41,     5,     5,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    41,     1,    58,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    41,     1,    58,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,     5,     1,    58,     5,     5,    44,     5,     5,     5,
       5,    44,    41,     5,     5,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    41,     1,    58,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    41,     1,    58,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,     4,     1,    58,
      44,     5,     4,   642,   145,    -1,   779,    -1,    41,    -1,
      -1,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    41,     1,    58,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    41,     1,
      58,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,     1,    -1,    58,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,     1,    -1,    -1,    -1,    -1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,     1,
      -1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,    47,
      48,    49,    50,    41,     1,    -1,    -1,    45,    46,    47,
      48,    49,    50,    -1,    41,     1,    -1,    -1,    45,    46,
      47,    48,    49,    50,    -1,     1,    -1,    -1,    -1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,     1,
      -1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,     1,    41,    -1,    -1,    -1,    45,
      46,    47,    48,    49,    50,    41,     1,    -1,    -1,    45,
      46,    47,    48,    49,    50,    -1,     1,    -1,    -1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,     1,
      -1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,     1,    -1,    41,    -1,    -1,    -1,
      45,    46,    47,    48,    49,    50,    41,     1,    -1,    -1,
      45,    46,    47,    48,    49,    50,    -1,     1,    -1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,     1,
      -1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,     1,    -1,    -1,    41,    -1,    -1,
      -1,    45,    46,    47,    48,    49,    50,    41,     1,    -1,
      -1,    45,    46,    47,    48,    49,    50,    -1,     1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,     1,
      -1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,     1,    -1,    -1,    -1,    41,    -1,
      -1,    -1,    45,    46,    47,    48,    49,    50,    41,     1,
      -1,    -1,    45,    46,    47,    48,    49,    50,    -1,    41,
       1,    -1,    -1,    45,    46,    47,    48,    49,    50,    -1,
       1,    -1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,     1,    -1,    -1,    -1,    -1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,     1,
      41,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
      41,     1,    -1,    -1,    45,    46,    47,    48,    49,    50,
      -1,     1,    -1,    -1,    41,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,     1,    -1,    -1,    -1,    -1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,     1,
      -1,    41,    -1,    -1,    -1,    45,    46,    47,    48,    49,
      50,    41,     1,    -1,    -1,    45,    46,    47,    48,    49,
      50,    -1,    -1,    -1,    41,     4,    -1,    -1,    45,    46,
      47,    48,    49,    50,    -1,    -1,    -1,    -1,    -1,    41,
      -1,    -1,    -1,    45,    46,    47,    48,    49,    50,    -1,
      -1,    -1,    41,    -1,    -1,    -1,    45,    46,    47,    48,
      49,    50,    41,    -1,    -1,    -1,    45,    46,    47,    48,
      49,    50,    41,    -1,    -1,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,   572,   573,   574,
     575,   576,   577,   578,   579,   580,   581
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   162,     0,     1,     3,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    43,    56,
      57,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
     148,   149,   150,   151,   152,   153,   154,   155,   156,   157,
     158,   159,   160,   163,   173,   176,   178,   179,   180,   181,
     182,   183,   184,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   241,
     242,   243,   244,   245,   246,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
       3,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,    41,     4,     4,     4,     4,    52,     4,
      41,    45,    46,    47,    48,    49,    50,   166,   167,   166,
      41,    44,    52,    53,    58,    44,    52,    52,    52,    41,
     174,    41,   175,    41,    41,     6,     1,   166,     1,   166,
       1,    44,    51,    52,    53,    54,    55,    58,   164,   165,
     166,   169,   170,     1,   164,     1,   164,     1,   164,     1,
     164,     1,   164,     1,   166,     1,   166,     1,   166,     1,
     166,     1,   166,     1,   166,     1,   166,     1,   166,     1,
     166,     1,   164,     1,   164,     1,     1,   164,     1,     1,
      44,     1,   164,     1,   164,     1,   166,     1,   166,     1,
     166,     1,   166,     1,   166,     1,   166,     1,   166,     1,
     166,     1,   166,     1,   166,     1,   166,     1,     4,    44,
       1,     4,    44,     1,     4,    44,     1,    44,     1,   166,
       1,    44,     1,    44,     1,   165,     1,   165,     1,     4,
      44,     1,     4,    44,     1,     4,    44,     1,   164,     1,
     164,     1,   164,     1,   164,     1,   164,     1,   164,     1,
     164,     1,   166,     1,    44,     1,    44,     1,    44,     1,
      44,     1,    44,     1,     1,     4,    44,     1,     4,    44,
       1,    44,     1,   164,     1,   166,     1,     4,    44,     1,
      41,    44,     1,    41,    44,     1,    41,    44,     1,    41,
      44,     1,   166,     1,   164,     1,   164,     1,   166,     1,
     166,     1,   166,     1,   166,     1,   166,     1,   166,     1,
     166,     1,   166,     1,   164,     1,   164,     1,   164,     1,
       1,   166,     1,   166,     1,   166,     1,   164,     1,   164,
       1,   164,     1,   164,     1,   164,     1,   164,     1,   164,
       1,    52,    53,   166,   171,     1,   166,   171,     1,   164,
       1,   164,     1,   166,     1,   171,     3,     3,   178,     3,
      44,   177,    58,   168,   169,   168,   177,   177,   177,    42,
     177,    41,   168,    41,   177,    42,   177,   177,     5,     5,
       5,     5,     6,     5,     5,     5,     5,     5,     4,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     3,    42,   177,    42,    42,    42,
      42,     4,    52,     4,    44,     4,    44,     4,     4,    44,
       4,     4,     4,    41,    41,   177,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   165,     4,    44,
     166,   170,   166,   170,   166,   170,   166,   170,   166,   170,
     166,   170,   166,   170,   166,   170,   166,   170,   166,   170,
     164,   164,     4,    44,   165,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   166,   166,     4,   166,     4,
     166,   166,   166,   164,   164,   164,    41,    44,    41,    44,
      41,    44,    41,    44,    41,    44,    41,    44,   165,   164,
     164,   165,   165,   165,    44,   166,    44,   166,    44,   166,
      44,   166,   165,   165,   165,   164,   164,   164,   164,   164,
     164,   164,   164,   164,    42,     4,    52,   172,   172,     5,
      42,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       4,   166,   166,   164,   164,   164,   164,   166,   166,   164,
     164,   166,   166,   164,   164,    44,    44,   168,    44,   247,
     247,     5,     5,     4,    44,     5,     4
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
#line 242 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax, LINE - 1 ); }
    break;

  case 35:
#line 258 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addEntry(); }
    break;

  case 36:
#line 259 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->setModuleName( (yyvsp[(2) - (2)]) ); }
    break;

  case 37:
#line 260 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 38:
#line 261 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 39:
#line 262 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 40:
#line 263 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 41:
#line 264 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 42:
#line 265 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 43:
#line 266 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 44:
#line 267 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 45:
#line 268 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLocal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 46:
#line 269 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addParam( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 47:
#line 270 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 48:
#line 271 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 49:
#line 272 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 50:
#line 273 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 51:
#line 274 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); }
    break;

  case 52:
#line 275 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]), false ); }
    break;

  case 53:
#line 276 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]), true ); }
    break;

  case 54:
#line 277 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 55:
#line 278 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]), 0, false ); }
    break;

  case 56:
#line 279 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addImport( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]), 0, true ); }
    break;

  case 57:
#line 280 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addImport( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), false );
   }
    break;

  case 58:
#line 283 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {
      COMPILER->addImport( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), (yyvsp[(5) - (5)]), true );
   }
    break;

  case 59:
#line 286 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 60:
#line 287 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 61:
#line 288 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 62:
#line 289 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 63:
#line 290 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 64:
#line 291 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 65:
#line 292 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 66:
#line 293 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 67:
#line 294 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 68:
#line 295 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (6)]), (yyvsp[(6) - (6)]), (yyvsp[(4) - (6)]) ); }
    break;

  case 69:
#line 296 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDEndSwitch(); }
    break;

  case 70:
#line 297 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 71:
#line 298 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 72:
#line 299 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addPropRef( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 75:
#line 302 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 76:
#line 303 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 77:
#line 304 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 78:
#line 305 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 79:
#line 306 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 80:
#line 307 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 81:
#line 308 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassCtor( (yyvsp[(2) - (2)]) ); }
    break;

  case 82:
#line 309 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); /* Currently the same as .endfunc */ }
    break;

  case 83:
#line 310 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInherit((yyvsp[(2) - (2)])); }
    break;

  case 84:
#line 311 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFrom( (yyvsp[(2) - (2)]) ); }
    break;

  case 85:
#line 312 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addExtern( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 86:
#line 313 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addDLine( (yyvsp[(2) - (2)]) ); }
    break;

  case 87:
#line 315 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 88:
#line 320 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         (yyvsp[(2) - (2)])->asString().exported( true );
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 89:
#line 326 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 90:
#line 334 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(1) - (1)]) ); }
    break;

  case 91:
#line 335 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(3) - (3)]) ); }
    break;

  case 92:
#line 339 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(1) - (1)]) ); }
    break;

  case 93:
#line 340 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(3) - (3)]) ); }
    break;

  case 94:
#line 343 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->defineLabel( (yyvsp[(1) - (2)])->asLabel() ); }
    break;

  case 95:
#line 347 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {(yyval) = new Falcon::Pseudo( LINE, (Falcon::int64) 0 ); }
    break;

  case 200:
#line 458 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 201:
#line 459 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LD" ); }
    break;

  case 202:
#line 463 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDRF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 203:
#line 464 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDRF" ); }
    break;

  case 204:
#line 468 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LNIL, (yyvsp[(2) - (2)]) ); }
    break;

  case 205:
#line 469 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LNIL" ); }
    break;

  case 206:
#line 473 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 207:
#line 474 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADD" ); }
    break;

  case 208:
#line 478 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 209:
#line 479 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADDS" ); }
    break;

  case 210:
#line 484 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 211:
#line 485 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUB" ); }
    break;

  case 212:
#line 489 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUBS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 213:
#line 490 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUBS" ); }
    break;

  case 214:
#line 494 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MUL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 215:
#line 495 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MUL" ); }
    break;

  case 216:
#line 499 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MULS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 217:
#line 500 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MULS" ); }
    break;

  case 218:
#line 505 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 219:
#line 506 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIV" ); }
    break;

  case 220:
#line 510 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 221:
#line 511 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIVS" ); }
    break;

  case 222:
#line 515 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MOD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 223:
#line 516 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MOD" ); }
    break;

  case 224:
#line 520 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POW, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 225:
#line 521 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POW" ); }
    break;

  case 226:
#line 526 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 227:
#line 527 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EQ" ); }
    break;

  case 228:
#line 531 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 229:
#line 532 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEQ" ); }
    break;

  case 230:
#line 536 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 231:
#line 537 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GE" ); }
    break;

  case 232:
#line 541 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 233:
#line 542 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GT" ); }
    break;

  case 234:
#line 546 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 235:
#line 547 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LE" ); }
    break;

  case 236:
#line 551 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 237:
#line 552 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LT" ); }
    break;

  case 238:
#line 556 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 239:
#line 557 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 240:
#line 558 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRY" ); }
    break;

  case 241:
#line 562 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INC, (yyvsp[(2) - (2)]) ); }
    break;

  case 242:
#line 563 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INC" ); }
    break;

  case 243:
#line 567 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DEC, (yyvsp[(2) - (2)])  ); }
    break;

  case 244:
#line 568 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DEC" ); }
    break;

  case 245:
#line 573 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INCP, (yyvsp[(2) - (2)]) ); }
    break;

  case 246:
#line 574 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INCP" ); }
    break;

  case 247:
#line 578 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DECP, (yyvsp[(2) - (2)])  ); }
    break;

  case 248:
#line 579 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DECP" ); }
    break;

  case 249:
#line 584 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEG, (yyvsp[(2) - (2)])  ); }
    break;

  case 250:
#line 585 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEG" ); }
    break;

  case 251:
#line 589 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOT, (yyvsp[(2) - (2)])  ); }
    break;

  case 252:
#line 590 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOT" ); }
    break;

  case 253:
#line 594 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 254:
#line 595 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 255:
#line 596 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "CALL" ); }
    break;

  case 256:
#line 600 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 257:
#line 601 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 258:
#line 602 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INST" ); }
    break;

  case 259:
#line 606 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_UNPK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 260:
#line 607 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPK" ); }
    break;

  case 261:
#line 611 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_UNPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 262:
#line 612 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPS" ); }
    break;

  case 263:
#line 617 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PUSH, (yyvsp[(2) - (2)]) ); }
    break;

  case 264:
#line 618 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHN ); }
    break;

  case 265:
#line 619 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PUSH" ); }
    break;

  case 266:
#line 623 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHR, (yyvsp[(2) - (2)]) ); }
    break;

  case 267:
#line 624 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSHR" ); }
    break;

  case 268:
#line 629 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_POP, (yyvsp[(2) - (2)]) ); }
    break;

  case 269:
#line 630 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POP" ); }
    break;

  case 270:
#line 634 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PEEK, (yyvsp[(2) - (2)]) ); }
    break;

  case 271:
#line 635 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PEEK" ); }
    break;

  case 272:
#line 639 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 273:
#line 640 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XPOP" ); }
    break;

  case 274:
#line 645 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 275:
#line 646 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 276:
#line 647 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDV" ); }
    break;

  case 277:
#line 651 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 278:
#line 652 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 279:
#line 653 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVT" ); }
    break;

  case 280:
#line 657 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 281:
#line 658 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 282:
#line 659 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STV" ); }
    break;

  case 283:
#line 663 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 284:
#line 664 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 285:
#line 665 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVR" ); }
    break;

  case 286:
#line 669 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 287:
#line 670 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 288:
#line 671 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVS" ); }
    break;

  case 289:
#line 675 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 290:
#line 676 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 291:
#line 677 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDP" ); yyerrok; }
    break;

  case 292:
#line 681 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 293:
#line 682 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 294:
#line 683 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPT" ); yyerrok; }
    break;

  case 295:
#line 687 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 296:
#line 688 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 297:
#line 689 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STP" ); }
    break;

  case 298:
#line 693 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 299:
#line 694 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 300:
#line 695 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPR" ); }
    break;

  case 301:
#line 699 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 302:
#line 700 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 303:
#line 701 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPS" ); }
    break;

  case 304:
#line 705 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 305:
#line 706 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 306:
#line 707 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAV" ); }
    break;

  case 307:
#line 711 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 308:
#line 712 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 309:
#line 713 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAN" ); }
    break;

  case 310:
#line 717 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 311:
#line 718 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 312:
#line 719 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAL" ); }
    break;

  case 313:
#line 723 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_IPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 314:
#line 724 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IPOP" ); }
    break;

  case 315:
#line 728 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GENA, (yyvsp[(2) - (2)]) ); }
    break;

  case 316:
#line 729 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENA" ); }
    break;

  case 317:
#line 733 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GEND, (yyvsp[(2) - (2)]) ); }
    break;

  case 318:
#line 734 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEND" ); }
    break;

  case 319:
#line 738 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GENR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 320:
#line 739 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENR" ); }
    break;

  case 321:
#line 743 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GEOR, (yyvsp[(2) - (2)]) ); }
    break;

  case 322:
#line 744 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEOR" ); }
    break;

  case 323:
#line 748 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RIS, (yyvsp[(2) - (2)]) ); }
    break;

  case 324:
#line 749 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RIS" ); }
    break;

  case 325:
#line 753 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 326:
#line 754 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 327:
#line 755 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JMP" ); }
    break;

  case 328:
#line 759 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOOL, (yyvsp[(1) - (2)]) ); }
    break;

  case 329:
#line 760 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOOL" ); }
    break;

  case 330:
#line 764 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 331:
#line 765 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 332:
#line 766 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFT" ); }
    break;

  case 333:
#line 770 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 334:
#line 771 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 335:
#line 772 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFF" ); }
    break;

  case 336:
#line 777 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 337:
#line 778 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 338:
#line 779 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORK" ); }
    break;

  case 339:
#line 783 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 340:
#line 784 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 341:
#line 785 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JTRY" ); }
    break;

  case 342:
#line 789 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RET ); }
    break;

  case 343:
#line 790 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RET" ); }
    break;

  case 344:
#line 794 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETA ); }
    break;

  case 345:
#line 795 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETA" ); }
    break;

  case 346:
#line 799 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETV, (yyvsp[(2) - (2)]) ); }
    break;

  case 347:
#line 800 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETV" ); }
    break;

  case 348:
#line 804 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOP ); }
    break;

  case 349:
#line 805 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOP" ); }
    break;

  case 350:
#line 809 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_PTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 351:
#line 810 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PTRY" ); }
    break;

  case 352:
#line 814 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_END ); }
    break;

  case 353:
#line 815 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "END" ); }
    break;

  case 354:
#line 819 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 355:
#line 820 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SWCH" ); }
    break;

  case 356:
#line 824 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 357:
#line 825 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SELE" ); }
    break;

  case 358:
#line 830 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {
         Falcon::Pseudo *psd = new Falcon::Pseudo( Falcon::Pseudo::tswitch_list );
         psd->line( LINE );
         psd->asList()->pushBack( (yyvsp[(1) - (3)]) );
         psd->asList()->pushBack( (yyvsp[(3) - (3)]) );
         (yyval) = psd;
      }
    break;

  case 359:
#line 839 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    {
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(3) - (5)]) );
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(5) - (5)]) );
         (yyval) = (yyvsp[(1) - (5)]);
      }
    break;

  case 360:
#line 847 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 361:
#line 848 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 362:
#line 849 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ONCE" ); }
    break;

  case 363:
#line 853 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 364:
#line 854 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 365:
#line 855 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 366:
#line 856 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 367:
#line 857 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BAND" ); }
    break;

  case 368:
#line 861 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 369:
#line 862 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 370:
#line 863 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 371:
#line 864 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 372:
#line 865 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOR" ); }
    break;

  case 373:
#line 869 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 374:
#line 870 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 375:
#line 871 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 376:
#line 872 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 377:
#line 873 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 378:
#line 877 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 379:
#line 878 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 380:
#line 879 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 381:
#line 883 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_AND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 382:
#line 884 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "AND" ); }
    break;

  case 383:
#line 888 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_OR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 384:
#line 889 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "OR" ); }
    break;

  case 385:
#line 893 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ANDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 386:
#line 894 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ANDS" ); }
    break;

  case 387:
#line 898 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 388:
#line 899 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ORS" ); }
    break;

  case 389:
#line 903 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 390:
#line 904 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XORS" ); }
    break;

  case 391:
#line 908 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MODS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 392:
#line 909 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MODS" ); }
    break;

  case 393:
#line 913 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POWS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 394:
#line 914 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POWS" ); }
    break;

  case 395:
#line 918 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOTS, (yyvsp[(2) - (2)]) ); }
    break;

  case 396:
#line 919 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOTS" ); }
    break;

  case 397:
#line 923 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 398:
#line 924 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 399:
#line 925 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HAS" ); }
    break;

  case 400:
#line 929 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 401:
#line 930 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 402:
#line 931 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HASN" ); }
    break;

  case 403:
#line 935 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 404:
#line 936 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 405:
#line 937 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVE" ); }
    break;

  case 406:
#line 941 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 407:
#line 942 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 408:
#line 943 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVN" ); }
    break;

  case 409:
#line 948 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_IN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 410:
#line 949 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IN" ); }
    break;

  case 411:
#line 953 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOIN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 412:
#line 954 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOIN" ); }
    break;

  case 413:
#line 958 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PROV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 414:
#line 959 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PROV" ); }
    break;

  case 415:
#line 963 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSIN, (yyvsp[(2) - (2)]) ); }
    break;

  case 416:
#line 964 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSIN" ); }
    break;

  case 417:
#line 968 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PASS, (yyvsp[(2) - (2)]) ); }
    break;

  case 418:
#line 969 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PASS" ); }
    break;

  case 419:
#line 973 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 420:
#line 974 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHR" ); }
    break;

  case 421:
#line 978 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 422:
#line 979 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHL" ); }
    break;

  case 423:
#line 983 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHRS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 424:
#line 984 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHRS" ); }
    break;

  case 425:
#line 988 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHLS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 426:
#line 989 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHLS" ); }
    break;

  case 427:
#line 993 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 428:
#line 994 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVR" ); }
    break;

  case 429:
#line 998 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 430:
#line 999 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPR" ); }
    break;

  case 431:
#line 1003 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LSB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 432:
#line 1004 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LSB" ); }
    break;

  case 433:
#line 1008 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 434:
#line 1009 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 435:
#line 1010 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INDI" ); }
    break;

  case 436:
#line 1014 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 437:
#line 1015 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 438:
#line 1016 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "STEX" ); }
    break;

  case 439:
#line 1020 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_TRAC, (yyvsp[(2) - (2)]) ); }
    break;

  case 440:
#line 1021 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "TRAC" ); }
    break;

  case 441:
#line 1025 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_WRT, (yyvsp[(2) - (2)]) ); }
    break;

  case 442:
#line 1026 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "WRT" ); }
    break;

  case 443:
#line 1031 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STO, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 444:
#line 1032 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STO" ); }
    break;

  case 445:
#line 1036 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_FORB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 446:
#line 1037 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORB" ); }
    break;


/* Line 1267 of yacc.c.  */
#line 4240 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.cpp"
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


#line 1040 "/export/medusa/gniccola/falcon/core/engine/fasm_parser.yy"
 /* c code */


/****************************************************
* C Code for falcon HSM compiler
*****************************************************/


void fasm_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of falcon_parser.yxx */


