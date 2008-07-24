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
     DSWITCH = 283,
     DSELECT = 284,
     DCASE = 285,
     DENDSWITCH = 286,
     DLINE = 287,
     DSTRING = 288,
     DISTRING = 289,
     DCSTRING = 290,
     DHAS = 291,
     DHASNT = 292,
     DINHERIT = 293,
     DINSTANCE = 294,
     SYMBOL = 295,
     EXPORT = 296,
     LABEL = 297,
     INTEGER = 298,
     REG_A = 299,
     REG_B = 300,
     REG_S1 = 301,
     REG_S2 = 302,
     REG_L1 = 303,
     REG_L2 = 304,
     NUMERIC = 305,
     STRING = 306,
     STRING_ID = 307,
     TRUE_TOKEN = 308,
     FALSE_TOKEN = 309,
     I_LD = 310,
     I_LNIL = 311,
     NIL = 312,
     I_ADD = 313,
     I_SUB = 314,
     I_MUL = 315,
     I_DIV = 316,
     I_MOD = 317,
     I_POW = 318,
     I_ADDS = 319,
     I_SUBS = 320,
     I_MULS = 321,
     I_DIVS = 322,
     I_POWS = 323,
     I_INC = 324,
     I_DEC = 325,
     I_INCP = 326,
     I_DECP = 327,
     I_NEG = 328,
     I_NOT = 329,
     I_RET = 330,
     I_RETV = 331,
     I_RETA = 332,
     I_FORK = 333,
     I_PUSH = 334,
     I_PSHR = 335,
     I_PSHN = 336,
     I_POP = 337,
     I_LDV = 338,
     I_LDVT = 339,
     I_STV = 340,
     I_STVR = 341,
     I_STVS = 342,
     I_LDP = 343,
     I_LDPT = 344,
     I_STP = 345,
     I_STPR = 346,
     I_STPS = 347,
     I_TRAV = 348,
     I_TRAN = 349,
     I_TRAL = 350,
     I_IPOP = 351,
     I_XPOP = 352,
     I_GENA = 353,
     I_GEND = 354,
     I_GENR = 355,
     I_GEOR = 356,
     I_JMP = 357,
     I_IFT = 358,
     I_IFF = 359,
     I_BOOL = 360,
     I_EQ = 361,
     I_NEQ = 362,
     I_GT = 363,
     I_GE = 364,
     I_LT = 365,
     I_LE = 366,
     I_UNPK = 367,
     I_UNPS = 368,
     I_CALL = 369,
     I_INST = 370,
     I_SWCH = 371,
     I_SELE = 372,
     I_NOP = 373,
     I_TRY = 374,
     I_JTRY = 375,
     I_PTRY = 376,
     I_RIS = 377,
     I_LDRF = 378,
     I_ONCE = 379,
     I_BAND = 380,
     I_BOR = 381,
     I_BXOR = 382,
     I_BNOT = 383,
     I_MODS = 384,
     I_AND = 385,
     I_OR = 386,
     I_ANDS = 387,
     I_ORS = 388,
     I_XORS = 389,
     I_NOTS = 390,
     I_HAS = 391,
     I_HASN = 392,
     I_GIVE = 393,
     I_GIVN = 394,
     I_IN = 395,
     I_NOIN = 396,
     I_PROV = 397,
     I_END = 398,
     I_PEEK = 399,
     I_PSIN = 400,
     I_PASS = 401,
     I_SHR = 402,
     I_SHL = 403,
     I_SHRS = 404,
     I_SHLS = 405,
     I_LDVR = 406,
     I_LDPR = 407,
     I_LSB = 408,
     I_INDI = 409,
     I_STEX = 410,
     I_TRAC = 411,
     I_WRT = 412,
     I_STO = 413
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
#define DSWITCH 283
#define DSELECT 284
#define DCASE 285
#define DENDSWITCH 286
#define DLINE 287
#define DSTRING 288
#define DISTRING 289
#define DCSTRING 290
#define DHAS 291
#define DHASNT 292
#define DINHERIT 293
#define DINSTANCE 294
#define SYMBOL 295
#define EXPORT 296
#define LABEL 297
#define INTEGER 298
#define REG_A 299
#define REG_B 300
#define REG_S1 301
#define REG_S2 302
#define REG_L1 303
#define REG_L2 304
#define NUMERIC 305
#define STRING 306
#define STRING_ID 307
#define TRUE_TOKEN 308
#define FALSE_TOKEN 309
#define I_LD 310
#define I_LNIL 311
#define NIL 312
#define I_ADD 313
#define I_SUB 314
#define I_MUL 315
#define I_DIV 316
#define I_MOD 317
#define I_POW 318
#define I_ADDS 319
#define I_SUBS 320
#define I_MULS 321
#define I_DIVS 322
#define I_POWS 323
#define I_INC 324
#define I_DEC 325
#define I_INCP 326
#define I_DECP 327
#define I_NEG 328
#define I_NOT 329
#define I_RET 330
#define I_RETV 331
#define I_RETA 332
#define I_FORK 333
#define I_PUSH 334
#define I_PSHR 335
#define I_PSHN 336
#define I_POP 337
#define I_LDV 338
#define I_LDVT 339
#define I_STV 340
#define I_STVR 341
#define I_STVS 342
#define I_LDP 343
#define I_LDPT 344
#define I_STP 345
#define I_STPR 346
#define I_STPS 347
#define I_TRAV 348
#define I_TRAN 349
#define I_TRAL 350
#define I_IPOP 351
#define I_XPOP 352
#define I_GENA 353
#define I_GEND 354
#define I_GENR 355
#define I_GEOR 356
#define I_JMP 357
#define I_IFT 358
#define I_IFF 359
#define I_BOOL 360
#define I_EQ 361
#define I_NEQ 362
#define I_GT 363
#define I_GE 364
#define I_LT 365
#define I_LE 366
#define I_UNPK 367
#define I_UNPS 368
#define I_CALL 369
#define I_INST 370
#define I_SWCH 371
#define I_SELE 372
#define I_NOP 373
#define I_TRY 374
#define I_JTRY 375
#define I_PTRY 376
#define I_RIS 377
#define I_LDRF 378
#define I_ONCE 379
#define I_BAND 380
#define I_BOR 381
#define I_BXOR 382
#define I_BNOT 383
#define I_MODS 384
#define I_AND 385
#define I_OR 386
#define I_ANDS 387
#define I_ORS 388
#define I_XORS 389
#define I_NOTS 390
#define I_HAS 391
#define I_HASN 392
#define I_GIVE 393
#define I_GIVN 394
#define I_IN 395
#define I_NOIN 396
#define I_PROV 397
#define I_END 398
#define I_PEEK 399
#define I_PSIN 400
#define I_PASS 401
#define I_SHR 402
#define I_SHL 403
#define I_SHRS 404
#define I_SHLS 405
#define I_LDVR 406
#define I_LDPR 407
#define I_LSB 408
#define I_INDI 409
#define I_STEX 410
#define I_TRAC 411
#define I_WRT 412
#define I_STO 413




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
#line 468 "/home/user/Progetti/falcon/core/engine/fasm_parser.cpp"

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
#define YYLAST   1812

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  159
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  123
/* YYNRULES -- Number of rules.  */
#define YYNRULES  441
/* YYNRULES -- Number of states.  */
#define YYNSTATES  798

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   413

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
     155,   156,   157,   158
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,     9,    12,    16,    19,    22,
      25,    27,    29,    31,    33,    35,    37,    39,    41,    43,
      45,    47,    49,    51,    53,    55,    57,    59,    61,    63,
      65,    67,    69,    71,    73,    76,    80,    85,    90,    96,
     100,   105,   109,   114,   118,   122,   125,   129,   133,   138,
     140,   143,   148,   153,   158,   163,   168,   173,   178,   183,
     188,   195,   197,   201,   205,   209,   212,   215,   220,   226,
     230,   235,   238,   242,   245,   247,   248,   253,   256,   260,
     263,   266,   269,   272,   274,   278,   280,   284,   287,   288,
     290,   294,   296,   298,   299,   301,   303,   305,   307,   309,
     311,   313,   315,   317,   319,   321,   323,   325,   327,   329,
     331,   333,   335,   337,   339,   341,   343,   345,   347,   349,
     351,   353,   355,   357,   359,   361,   363,   365,   367,   369,
     371,   373,   375,   377,   379,   381,   383,   385,   387,   389,
     391,   393,   395,   397,   399,   401,   403,   405,   407,   409,
     411,   413,   415,   417,   419,   421,   423,   425,   427,   429,
     431,   433,   435,   437,   439,   441,   443,   445,   447,   449,
     451,   453,   455,   457,   459,   461,   463,   465,   467,   469,
     471,   473,   475,   477,   479,   481,   483,   485,   487,   489,
     491,   493,   495,   497,   499,   501,   503,   505,   510,   513,
     518,   521,   524,   527,   532,   535,   540,   543,   548,   551,
     556,   559,   564,   567,   572,   575,   580,   583,   588,   591,
     596,   599,   604,   607,   612,   615,   620,   623,   628,   631,
     636,   639,   644,   647,   652,   655,   658,   661,   664,   667,
     670,   673,   676,   679,   682,   685,   688,   691,   694,   697,
     700,   705,   710,   713,   718,   723,   726,   731,   734,   739,
     742,   745,   747,   750,   753,   756,   759,   762,   765,   768,
     771,   774,   779,   784,   787,   794,   801,   804,   811,   818,
     821,   828,   835,   838,   843,   848,   851,   856,   861,   864,
     871,   878,   881,   888,   895,   898,   905,   912,   915,   920,
     925,   928,   935,   942,   945,   952,   959,   962,   965,   968,
     971,   974,   977,   980,   983,   986,   989,   996,   999,  1002,
    1005,  1008,  1011,  1014,  1017,  1020,  1023,  1026,  1031,  1036,
    1039,  1044,  1049,  1052,  1057,  1062,  1065,  1068,  1071,  1074,
    1076,  1079,  1081,  1084,  1087,  1090,  1092,  1095,  1098,  1101,
    1103,  1106,  1113,  1116,  1123,  1126,  1130,  1136,  1141,  1146,
    1149,  1154,  1159,  1164,  1169,  1172,  1177,  1182,  1187,  1192,
    1195,  1200,  1205,  1210,  1215,  1218,  1221,  1224,  1227,  1232,
    1235,  1240,  1243,  1248,  1251,  1256,  1259,  1264,  1267,  1272,
    1275,  1280,  1283,  1286,  1289,  1294,  1299,  1302,  1307,  1312,
    1315,  1320,  1325,  1328,  1333,  1338,  1341,  1346,  1349,  1354,
    1357,  1362,  1365,  1368,  1371,  1374,  1377,  1382,  1385,  1390,
    1393,  1398,  1401,  1406,  1409,  1414,  1417,  1422,  1425,  1430,
    1433,  1436,  1439,  1442,  1445,  1448,  1451,  1454,  1457,  1460,
    1463,  1468
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     160,     0,    -1,    -1,   160,   161,    -1,     3,    -1,   170,
       3,    -1,   174,   178,     3,    -1,   174,     3,    -1,   178,
       3,    -1,     1,     3,    -1,    57,    -1,   163,    -1,   164,
      -1,   167,    -1,    40,    -1,   165,    -1,    44,    -1,    45,
      -1,    46,    -1,    47,    -1,    48,    -1,    49,    -1,    57,
      -1,   167,    -1,    50,    -1,    53,    -1,    54,    -1,   168,
      -1,    52,    -1,    51,    -1,    43,    -1,    52,    -1,    51,
      -1,     7,    -1,    26,     4,    -1,     8,     4,   177,    -1,
       8,     4,   177,    41,    -1,     9,     4,   166,   177,    -1,
       9,     4,   166,   177,    41,    -1,    10,     4,   166,    -1,
      10,     4,   166,    41,    -1,    11,     4,   177,    -1,    11,
       4,   177,    41,    -1,    12,     4,   177,    -1,    13,     4,
     177,    -1,    14,     4,    -1,    14,     4,    41,    -1,    16,
       4,   177,    -1,    16,     4,   177,    41,    -1,    15,    -1,
      27,     4,    -1,    28,   164,     5,     4,    -1,    28,   164,
       5,    43,    -1,    29,   164,     5,     4,    -1,    29,   164,
       5,    43,    -1,    30,    57,     5,     4,    -1,    30,    43,
       5,     4,    -1,    30,    51,     5,     4,    -1,    30,    52,
       5,     4,    -1,    30,    40,     5,     4,    -1,    30,    43,
       6,    43,     5,     4,    -1,    31,    -1,    18,     4,   166,
      -1,    18,     4,    40,    -1,    19,     4,    40,    -1,    36,
     172,    -1,    37,   173,    -1,    39,    40,     4,   177,    -1,
      39,    40,     4,   177,    41,    -1,    20,     4,   177,    -1,
      20,     4,   177,    41,    -1,    21,     4,    -1,    21,     4,
      41,    -1,    22,    40,    -1,    23,    -1,    -1,    38,    40,
     171,   175,    -1,    24,     4,    -1,    25,     4,   177,    -1,
      32,    43,    -1,    33,    51,    -1,    34,    51,    -1,    35,
      51,    -1,    40,    -1,   172,     5,    40,    -1,    40,    -1,
     173,     5,    40,    -1,    42,     6,    -1,    -1,   176,    -1,
     175,     5,   176,    -1,   166,    -1,    40,    -1,    -1,    43,
      -1,   179,    -1,   181,    -1,   182,    -1,   184,    -1,   186,
      -1,   188,    -1,   190,    -1,   191,    -1,   183,    -1,   185,
      -1,   187,    -1,   189,    -1,   199,    -1,   200,    -1,   201,
      -1,   202,    -1,   192,    -1,   193,    -1,   194,    -1,   195,
      -1,   196,    -1,   197,    -1,   203,    -1,   204,    -1,   209,
      -1,   210,    -1,   211,    -1,   213,    -1,   214,    -1,   215,
      -1,   216,    -1,   217,    -1,   218,    -1,   219,    -1,   220,
      -1,   221,    -1,   223,    -1,   222,    -1,   224,    -1,   225,
      -1,   226,    -1,   227,    -1,   228,    -1,   229,    -1,   230,
      -1,   231,    -1,   233,    -1,   234,    -1,   235,    -1,   236,
      -1,   237,    -1,   207,    -1,   208,    -1,   205,    -1,   206,
      -1,   239,    -1,   241,    -1,   240,    -1,   242,    -1,   198,
      -1,   243,    -1,   238,    -1,   232,    -1,   245,    -1,   246,
      -1,   180,    -1,   244,    -1,   249,    -1,   250,    -1,   251,
      -1,   252,    -1,   253,    -1,   254,    -1,   255,    -1,   256,
      -1,   257,    -1,   260,    -1,   258,    -1,   259,    -1,   261,
      -1,   262,    -1,   263,    -1,   264,    -1,   265,    -1,   266,
      -1,   267,    -1,   248,    -1,   212,    -1,   268,    -1,   269,
      -1,   270,    -1,   271,    -1,   272,    -1,   273,    -1,   274,
      -1,   275,    -1,   276,    -1,   277,    -1,   278,    -1,   279,
      -1,   280,    -1,   281,    -1,    55,   164,     5,   162,    -1,
      55,     1,    -1,   123,   164,     5,   162,    -1,   123,     1,
      -1,    56,   164,    -1,    56,     1,    -1,    58,   162,     5,
     162,    -1,    58,     1,    -1,    64,   164,     5,   162,    -1,
      64,     1,    -1,    59,   162,     5,   162,    -1,    59,     1,
      -1,    65,   164,     5,   162,    -1,    65,     1,    -1,    60,
     162,     5,   162,    -1,    60,     1,    -1,    66,   164,     5,
     162,    -1,    66,     1,    -1,    61,   162,     5,   162,    -1,
      61,     1,    -1,    67,   164,     5,   162,    -1,    67,     1,
      -1,    62,   162,     5,   162,    -1,    62,     1,    -1,    63,
     162,     5,   162,    -1,    63,     1,    -1,   106,   162,     5,
     162,    -1,   106,     1,    -1,   107,   162,     5,   162,    -1,
     107,     1,    -1,   109,   162,     5,   162,    -1,   109,     1,
      -1,   108,   162,     5,   162,    -1,   108,     1,    -1,   111,
     162,     5,   162,    -1,   111,     1,    -1,   110,   162,     5,
     162,    -1,   110,     1,    -1,   119,     4,    -1,   119,    43,
      -1,   119,     1,    -1,    69,   164,    -1,    69,     1,    -1,
      70,   164,    -1,    70,     1,    -1,    71,   164,    -1,    71,
       1,    -1,    72,   164,    -1,    72,     1,    -1,    73,   162,
      -1,    73,     1,    -1,    74,   162,    -1,    74,     1,    -1,
     114,    43,     5,   164,    -1,   114,    43,     5,     4,    -1,
     114,     1,    -1,   115,    43,     5,   164,    -1,   115,    43,
       5,     4,    -1,   115,     1,    -1,   112,   164,     5,   164,
      -1,   112,     1,    -1,   113,    43,     5,   164,    -1,   113,
       1,    -1,    79,   162,    -1,    81,    -1,    79,     1,    -1,
      80,   162,    -1,    80,     1,    -1,    82,   164,    -1,    82,
       1,    -1,   144,   164,    -1,   144,     1,    -1,    97,   164,
      -1,    97,     1,    -1,    83,   164,     5,   164,    -1,    83,
     164,     5,   168,    -1,    83,     1,    -1,    84,   164,     5,
     164,     5,   164,    -1,    84,   164,     5,   168,     5,   164,
      -1,    84,     1,    -1,    85,   164,     5,   164,     5,   162,
      -1,    85,   164,     5,   168,     5,   162,    -1,    85,     1,
      -1,    86,   164,     5,   164,     5,   162,    -1,    86,   164,
       5,   168,     5,   162,    -1,    86,     1,    -1,    87,   164,
       5,   164,    -1,    87,   164,     5,   168,    -1,    87,     1,
      -1,    88,   164,     5,   164,    -1,    88,   164,     5,   168,
      -1,    88,     1,    -1,    89,   164,     5,   164,     5,   164,
      -1,    89,   164,     5,   168,     5,   164,    -1,    89,     1,
      -1,    90,   164,     5,   164,     5,   162,    -1,    90,   164,
       5,   168,     5,   162,    -1,    90,     1,    -1,    91,   164,
       5,   164,     5,   164,    -1,    91,   164,     5,   168,     5,
     164,    -1,    91,     1,    -1,    92,   164,     5,   164,    -1,
      92,   164,     5,   168,    -1,    92,     1,    -1,    93,    43,
       5,   162,     5,   162,    -1,    93,     4,     5,   162,     5,
     162,    -1,    93,     1,    -1,    94,     4,     5,     4,     5,
      43,    -1,    94,    43,     5,    43,     5,    43,    -1,    94,
       1,    -1,    95,    43,    -1,    95,     4,    -1,    95,     1,
      -1,    96,    43,    -1,    96,     1,    -1,    98,    43,    -1,
      98,     1,    -1,    99,    43,    -1,    99,     1,    -1,   100,
     163,     5,   163,     5,   166,    -1,   100,     1,    -1,   101,
     163,    -1,   101,     1,    -1,   122,   162,    -1,   122,     1,
      -1,   102,     4,    -1,   102,    43,    -1,   102,     1,    -1,
     105,   162,    -1,   105,     1,    -1,   103,     4,     5,   162,
      -1,   103,    43,     5,   162,    -1,   103,     1,    -1,   104,
       4,     5,   162,    -1,   104,    43,     5,   162,    -1,   104,
       1,    -1,    78,    43,     5,     4,    -1,    78,    43,     5,
      43,    -1,    78,     1,    -1,   120,     4,    -1,   120,    43,
      -1,   120,     1,    -1,    75,    -1,    75,     1,    -1,    77,
      -1,    77,     1,    -1,    76,   162,    -1,    76,     1,    -1,
     118,    -1,   118,     1,    -1,   121,    43,    -1,   121,     1,
      -1,   143,    -1,   143,     1,    -1,   116,    43,     5,   164,
       5,   247,    -1,   116,     1,    -1,   117,    43,     5,   164,
       5,   247,    -1,   117,     1,    -1,    43,     5,     4,    -1,
     247,     5,    43,     5,     4,    -1,   124,    43,     5,   162,
      -1,   124,     4,     5,   162,    -1,   124,     1,    -1,   125,
      40,     5,    40,    -1,   125,    40,     5,    43,    -1,   125,
      43,     5,    40,    -1,   125,    43,     5,    43,    -1,   125,
       1,    -1,   126,    40,     5,    40,    -1,   126,    40,     5,
      43,    -1,   126,    43,     5,    40,    -1,   126,    43,     5,
      43,    -1,   126,     1,    -1,   127,    40,     5,    40,    -1,
     127,    40,     5,    43,    -1,   127,    43,     5,    40,    -1,
     127,    43,     5,    43,    -1,   127,     1,    -1,   128,    40,
      -1,   128,    43,    -1,   128,     1,    -1,   130,   162,     5,
     162,    -1,   130,     1,    -1,   131,   162,     5,   162,    -1,
     131,     1,    -1,   132,   164,     5,   163,    -1,   132,     1,
      -1,   133,   164,     5,   163,    -1,   133,     1,    -1,   134,
     164,     5,   163,    -1,   134,     1,    -1,   129,   164,     5,
     163,    -1,   129,     1,    -1,    68,   164,     5,   163,    -1,
      68,     1,    -1,   135,   164,    -1,   135,     1,    -1,   136,
     164,     5,   164,    -1,   136,   164,     5,    43,    -1,   136,
       1,    -1,   137,   164,     5,   164,    -1,   137,   164,     5,
      43,    -1,   137,     1,    -1,   138,   164,     5,   164,    -1,
     138,   164,     5,    43,    -1,   138,     1,    -1,   139,   164,
       5,   164,    -1,   139,   164,     5,    43,    -1,   139,     1,
      -1,   140,   162,     5,   163,    -1,   140,     1,    -1,   141,
     162,     5,   163,    -1,   141,     1,    -1,   142,   162,     5,
     163,    -1,   142,     1,    -1,   145,   164,    -1,   145,     1,
      -1,   146,   164,    -1,   146,     1,    -1,   147,   162,     5,
     162,    -1,   147,     1,    -1,   148,   162,     5,   162,    -1,
     148,     1,    -1,   149,   162,     5,   162,    -1,   149,     1,
      -1,   150,   162,     5,   162,    -1,   150,     1,    -1,   151,
     162,     5,   162,    -1,   151,     1,    -1,   152,   162,     5,
     162,    -1,   152,     1,    -1,   153,   162,     5,   162,    -1,
     153,     1,    -1,   154,   169,    -1,   154,   164,    -1,   154,
       1,    -1,   155,   169,    -1,   155,   164,    -1,   155,     1,
      -1,   156,   162,    -1,   156,     1,    -1,   157,   162,    -1,
     157,     1,    -1,   158,   164,     5,   162,    -1,   158,     1,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   229,   229,   231,   235,   236,   237,   238,   239,   240,
     243,   243,   244,   244,   245,   245,   246,   246,   246,   246,
     246,   246,   248,   248,   249,   249,   249,   249,   250,   250,
     250,   251,   251,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   294,   295,   296,   296,   297,   298,   299,
     300,   305,   311,   320,   321,   325,   326,   329,   331,   333,
     334,   338,   339,   343,   344,   348,   349,   350,   351,   352,
     353,   354,   355,   356,   357,   358,   359,   360,   361,   362,
     363,   364,   365,   366,   367,   368,   369,   370,   371,   372,
     373,   374,   375,   376,   377,   378,   379,   380,   381,   382,
     383,   384,   385,   386,   387,   388,   389,   390,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   401,   402,
     403,   404,   405,   406,   407,   408,   409,   410,   411,   412,
     413,   414,   415,   416,   417,   418,   419,   420,   421,   422,
     423,   424,   425,   426,   427,   428,   429,   430,   431,   432,
     433,   434,   435,   436,   437,   438,   439,   440,   441,   442,
     443,   444,   445,   446,   447,   448,   449,   453,   454,   458,
     459,   463,   464,   468,   469,   473,   474,   479,   480,   484,
     485,   489,   490,   494,   495,   500,   501,   505,   506,   510,
     511,   515,   516,   521,   522,   526,   527,   531,   532,   536,
     537,   541,   542,   546,   547,   551,   552,   553,   557,   558,
     562,   563,   568,   569,   573,   574,   579,   580,   584,   585,
     589,   590,   591,   595,   596,   597,   601,   602,   606,   607,
     612,   613,   614,   618,   619,   624,   625,   629,   630,   634,
     635,   640,   641,   642,   646,   647,   648,   652,   653,   654,
     658,   659,   660,   664,   665,   666,   670,   671,   672,   676,
     677,   678,   682,   683,   684,   688,   689,   690,   694,   695,
     696,   700,   701,   702,   706,   707,   708,   712,   713,   714,
     718,   719,   723,   724,   728,   729,   733,   734,   738,   739,
     743,   744,   748,   749,   750,   754,   755,   759,   760,   761,
     765,   766,   767,   772,   773,   774,   778,   779,   780,   784,
     785,   789,   790,   794,   795,   799,   800,   804,   805,   809,
     810,   814,   815,   819,   820,   824,   833,   842,   843,   844,
     848,   849,   850,   851,   852,   856,   857,   858,   859,   860,
     864,   865,   866,   867,   868,   872,   873,   874,   878,   879,
     883,   884,   888,   889,   893,   894,   898,   899,   903,   904,
     908,   909,   913,   914,   918,   919,   920,   924,   925,   926,
     930,   931,   932,   936,   937,   938,   943,   944,   948,   949,
     953,   954,   958,   959,   963,   964,   968,   969,   973,   974,
     978,   979,   983,   984,   988,   989,   993,   994,   998,   999,
    1003,  1004,  1005,  1009,  1010,  1011,  1015,  1016,  1020,  1021,
    1026,  1027
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
  "DMODULE", "DLOAD", "DSWITCH", "DSELECT", "DCASE", "DENDSWITCH", "DLINE",
  "DSTRING", "DISTRING", "DCSTRING", "DHAS", "DHASNT", "DINHERIT",
  "DINSTANCE", "SYMBOL", "EXPORT", "LABEL", "INTEGER", "REG_A", "REG_B",
  "REG_S1", "REG_S2", "REG_L1", "REG_L2", "NUMERIC", "STRING", "STRING_ID",
  "TRUE_TOKEN", "FALSE_TOKEN", "I_LD", "I_LNIL", "NIL", "I_ADD", "I_SUB",
  "I_MUL", "I_DIV", "I_MOD", "I_POW", "I_ADDS", "I_SUBS", "I_MULS",
  "I_DIVS", "I_POWS", "I_INC", "I_DEC", "I_INCP", "I_DECP", "I_NEG",
  "I_NOT", "I_RET", "I_RETV", "I_RETA", "I_FORK", "I_PUSH", "I_PSHR",
  "I_PSHN", "I_POP", "I_LDV", "I_LDVT", "I_STV", "I_STVR", "I_STVS",
  "I_LDP", "I_LDPT", "I_STP", "I_STPR", "I_STPS", "I_TRAV", "I_TRAN",
  "I_TRAL", "I_IPOP", "I_XPOP", "I_GENA", "I_GEND", "I_GENR", "I_GEOR",
  "I_JMP", "I_IFT", "I_IFF", "I_BOOL", "I_EQ", "I_NEQ", "I_GT", "I_GE",
  "I_LT", "I_LE", "I_UNPK", "I_UNPS", "I_CALL", "I_INST", "I_SWCH",
  "I_SELE", "I_NOP", "I_TRY", "I_JTRY", "I_PTRY", "I_RIS", "I_LDRF",
  "I_ONCE", "I_BAND", "I_BOR", "I_BXOR", "I_BNOT", "I_MODS", "I_AND",
  "I_OR", "I_ANDS", "I_ORS", "I_XORS", "I_NOTS", "I_HAS", "I_HASN",
  "I_GIVE", "I_GIVN", "I_IN", "I_NOIN", "I_PROV", "I_END", "I_PEEK",
  "I_PSIN", "I_PASS", "I_SHR", "I_SHL", "I_SHRS", "I_SHLS", "I_LDVR",
  "I_LDPR", "I_LSB", "I_INDI", "I_STEX", "I_TRAC", "I_WRT", "I_STO",
  "$accept", "input", "line", "xoperand", "operand", "op_variable",
  "op_register", "x_op_immediate", "op_immediate", "op_scalar",
  "op_string", "directive", "@1", "has_symlist", "hasnt_symlist", "label",
  "inherit_param_list", "inherit_param", "def_line", "instruction",
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
  "inst_stex", "inst_trac", "inst_wrt", "inst_sto", 0
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
     405,   406,   407,   408,   409,   410,   411,   412,   413
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   159,   160,   160,   161,   161,   161,   161,   161,   161,
     162,   162,   163,   163,   164,   164,   165,   165,   165,   165,
     165,   165,   166,   166,   167,   167,   167,   167,   168,   168,
     168,   169,   169,   170,   170,   170,   170,   170,   170,   170,
     170,   170,   170,   170,   170,   170,   170,   170,   170,   170,
     170,   170,   170,   170,   170,   170,   170,   170,   170,   170,
     170,   170,   170,   170,   170,   170,   170,   170,   170,   170,
     170,   170,   170,   170,   170,   171,   170,   170,   170,   170,
     170,   170,   170,   172,   172,   173,   173,   174,   175,   175,
     175,   176,   176,   177,   177,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   179,   179,   180,
     180,   181,   181,   182,   182,   183,   183,   184,   184,   185,
     185,   186,   186,   187,   187,   188,   188,   189,   189,   190,
     190,   191,   191,   192,   192,   193,   193,   194,   194,   195,
     195,   196,   196,   197,   197,   198,   198,   198,   199,   199,
     200,   200,   201,   201,   202,   202,   203,   203,   204,   204,
     205,   205,   205,   206,   206,   206,   207,   207,   208,   208,
     209,   209,   209,   210,   210,   211,   211,   212,   212,   213,
     213,   214,   214,   214,   215,   215,   215,   216,   216,   216,
     217,   217,   217,   218,   218,   218,   219,   219,   219,   220,
     220,   220,   221,   221,   221,   222,   222,   222,   223,   223,
     223,   224,   224,   224,   225,   225,   225,   226,   226,   226,
     227,   227,   228,   228,   229,   229,   230,   230,   231,   231,
     232,   232,   233,   233,   233,   234,   234,   235,   235,   235,
     236,   236,   236,   237,   237,   237,   238,   238,   238,   239,
     239,   240,   240,   241,   241,   242,   242,   243,   243,   244,
     244,   245,   245,   246,   246,   247,   247,   248,   248,   248,
     249,   249,   249,   249,   249,   250,   250,   250,   250,   250,
     251,   251,   251,   251,   251,   252,   252,   252,   253,   253,
     254,   254,   255,   255,   256,   256,   257,   257,   258,   258,
     259,   259,   260,   260,   261,   261,   261,   262,   262,   262,
     263,   263,   263,   264,   264,   264,   265,   265,   266,   266,
     267,   267,   268,   268,   269,   269,   270,   270,   271,   271,
     272,   272,   273,   273,   274,   274,   275,   275,   276,   276,
     277,   277,   277,   278,   278,   278,   279,   279,   280,   280,
     281,   281
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     1,     2,     3,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     3,     4,     4,     5,     3,
       4,     3,     4,     3,     3,     2,     3,     3,     4,     1,
       2,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       6,     1,     3,     3,     3,     2,     2,     4,     5,     3,
       4,     2,     3,     2,     1,     0,     4,     2,     3,     2,
       2,     2,     2,     1,     3,     1,     3,     2,     0,     1,
       3,     1,     1,     0,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     4,     2,     4,
       2,     2,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       4,     4,     2,     4,     4,     2,     4,     2,     4,     2,
       2,     1,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     4,     4,     2,     6,     6,     2,     6,     6,     2,
       6,     6,     2,     4,     4,     2,     4,     4,     2,     6,
       6,     2,     6,     6,     2,     6,     6,     2,     4,     4,
       2,     6,     6,     2,     6,     6,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     6,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     4,     4,     2,
       4,     4,     2,     4,     4,     2,     2,     2,     2,     1,
       2,     1,     2,     2,     2,     1,     2,     2,     2,     1,
       2,     6,     2,     6,     2,     3,     5,     4,     4,     2,
       4,     4,     4,     4,     2,     4,     4,     4,     4,     2,
       4,     4,     4,     4,     2,     2,     2,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     2,     2,     4,     4,     2,     4,     4,     2,
       4,     4,     2,     4,     4,     2,     4,     2,     4,     2,
       4,     2,     2,     2,     2,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       4,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,     0,     4,    33,     0,     0,     0,     0,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
      74,     0,     0,     0,     0,     0,     0,     0,    61,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   261,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     3,     0,     0,     0,    95,   160,    96,    97,   103,
      98,   104,    99,   105,   100,   106,   101,   102,   111,   112,
     113,   114,   115,   116,   154,   107,   108,   109,   110,   117,
     118,   148,   149,   146,   147,   119,   120,   121,   182,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   132,   131,
     133,   134,   135,   136,   137,   138,   139,   140,   157,   141,
     142,   143,   144,   145,   156,   150,   152,   151,   153,   155,
     161,   158,   159,   181,   162,   163,   164,   165,   166,   167,
     168,   169,   170,   172,   173,   171,   174,   175,   176,   177,
     178,   179,   180,   183,   184,   185,   186,   187,   188,   189,
     190,   191,   192,   193,   194,   195,   196,     9,    93,     0,
       0,    93,    93,    93,    45,    93,     0,     0,    93,    71,
      73,    77,    93,    34,    50,    14,    16,    17,    18,    19,
      20,    21,     0,    15,     0,     0,     0,     0,     0,     0,
      79,    80,    81,    82,    83,    65,    85,    66,    75,     0,
      87,   198,     0,   202,   201,   204,    30,    24,    29,    28,
      25,    26,    10,     0,    11,    12,    13,    27,   208,     0,
     212,     0,   216,     0,   220,     0,   222,     0,   206,     0,
     210,     0,   214,     0,   218,     0,   391,     0,   239,   238,
     241,   240,   243,   242,   245,   244,   247,   246,   249,   248,
     340,   344,   343,   342,   335,     0,   262,   260,   264,   263,
     266,   265,   273,     0,   276,     0,   279,     0,   282,     0,
     285,     0,   288,     0,   291,     0,   294,     0,   297,     0,
     300,     0,   303,     0,     0,   306,     0,     0,   309,   308,
     307,   311,   310,   270,   269,   313,   312,   315,   314,   317,
       0,   319,   318,   324,   322,   323,   329,     0,     0,   332,
       0,     0,   326,   325,   224,     0,   226,     0,   230,     0,
     228,     0,   234,     0,   232,     0,   257,     0,   259,     0,
     252,     0,   255,     0,   352,     0,   354,     0,   346,   237,
     235,   236,   338,   336,   337,   348,   347,   321,   320,   200,
       0,   359,     0,     0,   364,     0,     0,   369,     0,     0,
     374,     0,     0,   377,   375,   376,   389,     0,   379,     0,
     381,     0,   383,     0,   385,     0,   387,     0,   393,   392,
     396,     0,   399,     0,   402,     0,   405,     0,   407,     0,
     409,     0,   411,     0,   350,   268,   267,   413,   412,   415,
     414,   417,     0,   419,     0,   421,     0,   423,     0,   425,
       0,   427,     0,   429,     0,   432,    32,    31,   431,   430,
     435,   434,   433,   437,   436,   439,   438,   441,     0,     5,
       7,     0,     8,    94,    35,    22,    93,    23,    39,    41,
      43,    44,    46,    47,    63,    62,    64,    69,    72,    78,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      88,    93,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     6,    36,    37,    40,
      42,    48,    70,    51,    52,    53,    54,    59,    56,     0,
      57,    58,    55,    84,    86,    92,    91,    76,    89,    67,
     197,   203,   207,   211,   215,   219,   221,   205,   209,   213,
     217,   390,   333,   334,   271,   272,     0,     0,     0,     0,
       0,     0,   283,   284,   286,   287,     0,     0,     0,     0,
       0,     0,   298,   299,     0,     0,     0,     0,     0,   327,
     328,   330,   331,   223,   225,   229,   227,   233,   231,   256,
     258,   251,   250,   254,   253,     0,     0,   199,   358,   357,
     360,   361,   362,   363,   365,   366,   367,   368,   370,   371,
     372,   373,   388,   378,   380,   382,   384,   386,   395,   394,
     398,   397,   401,   400,   404,   403,   406,   408,   410,   416,
     418,   420,   422,   424,   426,   428,   440,    38,     0,     0,
      68,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      60,    90,   274,   275,   277,   278,   280,   281,   289,   290,
     292,   293,   295,   296,   302,   301,   304,   305,   316,     0,
     351,   353,     0,     0,   355,     0,     0,   356
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,   141,   303,   304,   305,   273,   646,   306,   307,
     509,   142,   550,   285,   287,   143,   647,   648,   524,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   790,   213,
     214,   215,   216,   217,   218,   219,   220,   221,   222,   223,
     224,   225,   226,   227,   228,   229,   230,   231,   232,   233,
     234,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     244,   245,   246
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -246
static const yytype_int16 yypact[] =
{
    -246,   773,  -246,     3,  -246,  -246,     8,    32,    46,    84,
     105,   110,   161,  -246,   167,   168,   184,   185,   232,    47,
    -246,   235,   236,   282,   285,  1031,  1031,   117,  -246,   229,
     237,   239,   240,   253,   254,   256,   269,   304,    98,   627,
      83,   160,   176,   198,   214,   230,   722,  1316,  1331,  1345,
    1356,  1366,  1380,  1394,  1406,   268,   284,    80,   462,   181,
      18,   550,   652,  -246,  1416,  1429,  1443,  1456,  1466,  1478,
    1492,  1506,  1516,  1527,  1541,    29,    30,    31,    55,  1556,
     115,   121,   531,  1291,    74,    96,   457,   667,   707,   931,
     946,   961,   986,  1001,  1567,   147,   148,   297,   481,   484,
     231,   458,   475,   490,  1016,  1577,   477,    17,    36,   119,
     140,  1590,  1041,  1056,  1605,  1617,  1627,  1639,  1654,  1667,
    1677,  1688,  1071,  1096,  1111,   234,  1703,  1717,  1728,  1126,
    1151,  1166,  1181,  1206,  1221,  1236,    19,   255,  1261,  1276,
    1738,  -246,   320,   294,   452,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,   413,   583,
     583,   413,   413,   413,   419,   413,   414,   429,   413,   431,
    -246,  -246,   413,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,   465,  -246,   468,   469,    11,   470,   472,   479,
    -246,  -246,  -246,  -246,  -246,   482,  -246,   483,  -246,   476,
    -246,  -246,   485,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,   487,  -246,  -246,  -246,  -246,  -246,   498,
    -246,   516,  -246,   518,  -246,   526,  -246,   529,  -246,   533,
    -246,   534,  -246,   545,  -246,   547,  -246,   549,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,   551,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,   558,  -246,   559,  -246,   567,  -246,   568,
    -246,   586,  -246,   587,  -246,   601,  -246,   622,  -246,   624,
    -246,   625,  -246,   626,   636,  -246,   637,   649,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
     650,  -246,  -246,  -246,  -246,  -246,  -246,   651,   653,  -246,
     657,   658,  -246,  -246,  -246,   659,  -246,   660,  -246,   661,
    -246,   672,  -246,   673,  -246,   674,  -246,   675,  -246,   684,
    -246,   685,  -246,   686,  -246,   689,  -246,   717,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
     720,  -246,   724,   725,  -246,   729,   730,  -246,   738,   739,
    -246,   741,   743,  -246,  -246,  -246,  -246,   744,  -246,   758,
    -246,   760,  -246,   767,  -246,   770,  -246,   772,  -246,  -246,
    -246,   774,  -246,   785,  -246,   808,  -246,   809,  -246,   815,
    -246,   816,  -246,   819,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,   820,  -246,   825,  -246,   929,  -246,   930,  -246,
     938,  -246,   943,  -246,   967,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,   968,  -246,
    -246,   552,  -246,  -246,   445,  -246,   413,  -246,   591,   616,
    -246,  -246,  -246,   737,  -246,  -246,  -246,   987,  -246,  -246,
      34,   175,  1023,  1070,  1039,  1079,  1122,  1125,  1090,  1091,
     688,   413,   906,   906,   906,   906,   906,   906,   906,   906,
     906,   906,   906,  1748,   249,   893,   893,   893,   893,   893,
     893,   893,   893,   893,   893,   906,   906,  1128,  1092,  1748,
     906,   906,   906,   906,   906,   906,   906,   906,   906,   906,
    1031,  1031,   449,   513,  1031,  1031,   906,   906,   906,   -33,
      42,    70,   123,   130,   138,  1748,   906,   906,  1748,  1748,
    1748,   299,   565,   921,   976,  1748,  1748,  1748,   906,   906,
     906,   906,   906,   906,   906,   906,  -246,  -246,  1093,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  1132,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  1133,  -246,  1140,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  1179,  1180,  1182,  1183,
    1184,  1185,  -246,  -246,  -246,  -246,  1187,  1188,  1231,  1234,
    1235,  1237,  -246,  -246,  1239,  1240,  1242,  1243,  1286,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  1289,  1290,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  1129,   688,
    -246,  1031,  1031,   906,   906,   906,   906,  1031,  1031,   906,
     906,  1031,  1031,   906,   906,  1143,  1198,   583,  1200,  1200,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  1292,
    1293,  1293,  1295,  1253,  -246,  1297,  1296,  -246
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -246,  -246,  -246,    63,   -80,   -25,  -246,  -241,  -245,  1238,
     556,  -246,  -246,  -246,  -246,  -246,  -246,   554,  -200,  1204,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,   579,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -350
static const yytype_int16 yytable[] =
{
     272,   274,   390,   392,   527,   527,   247,   710,   526,   528,
     711,   527,   248,   292,   294,   535,   543,   544,   444,   344,
     505,   319,   321,   323,   325,   327,   329,   331,   333,   335,
     372,   375,   378,   373,   376,   379,   249,   447,   633,   351,
     353,   355,   357,   359,   361,   363,   365,   367,   369,   371,
     250,   529,   530,   531,   384,   533,   381,   445,   537,   265,
     446,   345,   539,   266,   267,   268,   269,   270,   271,   417,
     506,   507,   374,   377,   380,   393,   448,   634,   394,   449,
     440,   340,   712,  -339,   295,   713,   457,   260,   251,   463,
     465,   467,   469,   471,   473,   475,   477,   396,   382,   291,
     397,   486,   488,   490,   309,   311,   313,   315,   317,   252,
     714,   508,   511,   715,   253,   518,   385,   395,   337,   339,
     450,   342,   387,   265,   347,   349,   296,   266,   267,   268,
     269,   270,   271,   297,   298,   299,   300,   301,   265,   398,
     302,   453,   266,   267,   268,   269,   270,   271,   418,   420,
     403,   405,   407,   409,   411,   413,   415,   275,   386,   451,
     276,   308,   452,   716,   388,   254,   717,   438,   277,   278,
     718,   255,   256,   719,   279,   459,   461,   310,   720,   635,
     454,   721,   343,   455,  -341,   479,   481,   483,   257,   258,
     419,   421,   492,   494,   496,   498,   500,   502,   504,   312,
     265,   514,   516,   296,   266,   267,   268,   269,   270,   271,
     297,   298,   299,   300,   301,   314,   265,   302,   636,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   316,   428,   302,  -345,   484,   259,  -349,   265,   261,
     262,   296,   266,   267,   268,   269,   270,   271,   297,   298,
     299,   300,   301,   662,   265,   302,   510,   296,   266,   267,
     268,   269,   270,   271,   297,   298,   299,   300,   301,   336,
     265,   302,   280,   296,   266,   267,   268,   269,   270,   271,
     297,   298,   299,   300,   301,   338,   263,   302,   281,   264,
     282,   283,   663,   284,   286,   265,   288,   520,   422,   266,
     267,   268,   269,   270,   271,   527,   506,   507,   265,   289,
     290,   296,   266,   267,   268,   269,   270,   271,   297,   298,
     299,   300,   301,   519,   265,   302,   628,   296,   266,   267,
     268,   269,   270,   271,   297,   298,   299,   300,   301,   265,
     423,   302,   728,   266,   267,   268,   269,   270,   271,    38,
      39,   649,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   701,   534,   522,   523,   296,   399,   429,
     532,   400,   430,   341,   297,   298,   299,   300,   301,   536,
     540,   525,   538,   541,   542,   545,   432,   546,   441,   433,
     551,   442,   424,   661,   547,   426,   627,   548,   549,   265,
     552,   435,   553,   266,   267,   268,   269,   270,   271,   688,
     401,   431,   265,   554,   527,   296,   266,   267,   268,   269,
     270,   271,   297,   298,   299,   300,   301,   703,   434,   302,
     443,   555,   527,   556,   425,   722,   788,   427,   725,   726,
     727,   557,   389,   436,   558,   736,   737,   738,   559,   560,
     664,   666,   668,   670,   672,   674,   676,   678,   680,   682,
     561,   346,   562,   265,   563,   626,   564,   266,   267,   268,
     269,   270,   271,   565,   566,   699,   700,   702,   704,   705,
     706,   265,   567,   568,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   729,   731,   733,   735,
     265,   569,   570,   296,   266,   267,   268,   269,   270,   271,
     297,   298,   299,   300,   301,   265,   571,   302,   730,   266,
     267,   268,   269,   270,   271,   650,   651,   652,   653,   654,
     655,   656,   657,   658,   659,   660,   296,   572,   293,   573,
     574,   575,   629,   297,   298,   299,   300,   301,   684,   685,
     525,   576,   577,   689,   690,   691,   692,   693,   694,   695,
     696,   697,   698,   348,   578,   579,   580,   630,   581,   707,
     708,   709,   582,   583,   584,   585,   586,   265,   402,   723,
     724,   266,   267,   268,   269,   270,   271,   587,   588,   589,
     590,   739,   740,   741,   742,   743,   744,   745,   746,   591,
     592,   593,   265,   512,   594,   296,   266,   267,   268,   269,
     270,   271,   297,   298,   299,   300,   301,   265,   404,   302,
     296,   266,   267,   268,   269,   270,   271,   297,   298,   299,
     300,   301,   595,   318,   302,   596,   772,   773,   645,   597,
     598,   296,   778,   779,   599,   600,   782,   783,   297,   298,
     299,   300,   301,   601,   602,   525,   603,   265,   604,   605,
     296,   266,   267,   268,   269,   270,   271,   297,   298,   299,
     300,   301,   265,   606,   302,   607,   266,   267,   268,   269,
     270,   271,   608,     2,     3,   609,     4,   610,   631,   611,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
     612,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,   613,   614,    37,   774,   775,   776,   777,
     615,   616,   780,   781,   617,   618,   784,   785,    38,    39,
     619,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   406,   265,   620,   621,   296,   266,   267,   268,
     269,   270,   271,   622,   298,   299,   265,   408,   623,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   265,   410,   302,   732,   266,   267,   268,   269,   270,
     271,   265,   624,   625,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   265,   412,   302,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   265,   414,   302,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   265,   437,   302,   734,
     266,   267,   268,   269,   270,   271,   265,   637,   632,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   265,   458,   302,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   265,   460,   302,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   265,   478,   302,   638,   266,   267,   268,   269,   270,
     271,   265,   639,   640,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   265,   480,   302,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   265,   482,   302,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   641,   491,   302,   642,
     643,   644,   686,   770,   747,   687,   265,   748,   749,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   265,   493,   302,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   265,   495,   302,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   750,   497,   302,   751,   752,   786,   753,   754,   755,
     756,   265,   757,   758,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   265,   499,   302,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   265,   501,   302,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   759,   503,   302,   760,
     761,   787,   762,   789,   763,   764,   265,   765,   766,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   265,   513,   302,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   265,   515,   302,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   767,   391,   302,   768,   769,   795,   792,   793,   794,
     797,   265,   796,   771,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   265,   320,   302,   296,
     266,   267,   268,   269,   270,   271,   297,   298,   299,   300,
     301,   265,   322,   302,   296,   266,   267,   268,   269,   270,
     271,   297,   298,   299,   300,   301,   324,   521,   791,     0,
       0,     0,     0,     0,     0,     0,   265,   326,     0,     0,
     266,   267,   268,   269,   270,   271,     0,   328,     0,     0,
       0,   265,     0,     0,     0,   266,   267,   268,   269,   270,
     271,   330,     0,     0,     0,   265,     0,     0,     0,   266,
     267,   268,   269,   270,   271,   332,   265,     0,     0,     0,
     266,   267,   268,   269,   270,   271,   265,   334,     0,     0,
     266,   267,   268,   269,   270,   271,     0,   350,     0,     0,
     265,     0,     0,     0,   266,   267,   268,   269,   270,   271,
     352,     0,     0,     0,   265,     0,     0,     0,   266,   267,
     268,   269,   270,   271,   354,     0,   265,     0,     0,     0,
     266,   267,   268,   269,   270,   271,   265,   356,     0,     0,
     266,   267,   268,   269,   270,   271,     0,   358,     0,   265,
       0,     0,     0,   266,   267,   268,   269,   270,   271,   360,
       0,     0,     0,   265,     0,     0,     0,   266,   267,   268,
     269,   270,   271,   362,     0,     0,   265,     0,     0,     0,
     266,   267,   268,   269,   270,   271,   265,   364,     0,     0,
     266,   267,   268,   269,   270,   271,     0,   366,   265,     0,
       0,     0,   266,   267,   268,   269,   270,   271,   368,     0,
       0,     0,   265,     0,     0,     0,   266,   267,   268,   269,
     270,   271,   370,     0,     0,     0,   265,     0,     0,     0,
     266,   267,   268,   269,   270,   271,   265,   383,     0,     0,
     266,   267,   268,   269,   270,   271,     0,   265,   416,     0,
       0,   266,   267,   268,   269,   270,   271,     0,   439,     0,
       0,   265,     0,     0,     0,   266,   267,   268,   269,   270,
     271,   456,     0,     0,     0,     0,   265,     0,     0,     0,
     266,   267,   268,   269,   270,   271,   462,   265,     0,     0,
       0,   266,   267,   268,   269,   270,   271,   265,   464,     0,
       0,   266,   267,   268,   269,   270,   271,     0,   466,     0,
     265,     0,     0,     0,   266,   267,   268,   269,   270,   271,
     468,     0,     0,     0,     0,   265,     0,     0,     0,   266,
     267,   268,   269,   270,   271,   470,     0,   265,     0,     0,
       0,   266,   267,   268,   269,   270,   271,   265,   472,     0,
       0,   266,   267,   268,   269,   270,   271,     0,   474,   265,
       0,     0,     0,   266,   267,   268,   269,   270,   271,   476,
       0,     0,     0,     0,   265,     0,     0,     0,   266,   267,
     268,   269,   270,   271,   485,     0,     0,   265,     0,     0,
       0,   266,   267,   268,   269,   270,   271,   265,   487,     0,
       0,   266,   267,   268,   269,   270,   271,     0,   265,   489,
       0,     0,   266,   267,   268,   269,   270,   271,     0,   517,
       0,     0,     0,   265,     0,     0,     0,   266,   267,   268,
     269,   270,   271,     0,     0,     0,     0,   265,     0,     0,
       0,   266,   267,   268,   269,   270,   271,     0,   265,     0,
       0,     0,   266,   267,   268,   269,   270,   271,   265,     0,
       0,     0,   266,   267,   268,   269,   270,   271,   265,     0,
       0,   296,   266,   267,   268,   269,   270,   271,   297,   298,
     299,   300,   301,   665,   667,   669,   671,   673,   675,   677,
     679,   681,   683
};

static const yytype_int16 yycheck[] =
{
      25,    26,    82,    83,   249,   250,     3,    40,   249,   250,
      43,   256,     4,    38,    39,   256,     5,     6,     1,     1,
       1,    46,    47,    48,    49,    50,    51,    52,    53,    54,
       1,     1,     1,     4,     4,     4,     4,     1,     4,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
       4,   251,   252,   253,    79,   255,     1,    40,   258,    40,
      43,    43,   262,    44,    45,    46,    47,    48,    49,    94,
      51,    52,    43,    43,    43,     1,    40,    43,     4,    43,
     105,     1,    40,     3,     1,    43,   111,    40,     4,   114,
     115,   116,   117,   118,   119,   120,   121,     1,    43,     1,
       4,   126,   127,   128,    41,    42,    43,    44,    45,     4,
      40,   136,   137,    43,     4,   140,     1,    43,    55,    56,
       1,    58,     1,    40,    61,    62,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    40,    43,
      57,     1,    44,    45,    46,    47,    48,    49,     1,     1,
      87,    88,    89,    90,    91,    92,    93,    40,    43,    40,
      43,     1,    43,    40,    43,     4,    43,   104,    51,    52,
      40,     4,     4,    43,    57,   112,   113,     1,    40,     4,
      40,    43,     1,    43,     3,   122,   123,   124,     4,     4,
      43,    43,   129,   130,   131,   132,   133,   134,   135,     1,
      40,   138,   139,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,     1,    40,    57,    43,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,     1,     1,    57,     3,     1,     4,     3,    40,     4,
       4,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,     4,    40,    57,     1,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,     1,
      40,    57,    43,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,     1,     4,    57,    51,     4,
      51,    51,    43,    40,    40,    40,    40,     3,     1,    44,
      45,    46,    47,    48,    49,   550,    51,    52,    40,    40,
       6,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,     3,    40,    57,   526,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    40,
      43,    57,    43,    44,    45,    46,    47,    48,    49,    55,
      56,   551,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,   140,   141,   142,   143,   144,   145,
     146,   147,   148,   149,   150,   151,   152,   153,   154,   155,
     156,   157,   158,     4,    40,     3,    43,    43,     1,     1,
      41,     4,     4,     1,    50,    51,    52,    53,    54,    40,
       5,    57,    41,     5,     5,     5,     1,     5,     1,     4,
       4,     4,     1,   563,     5,     1,    41,     5,     5,    40,
       5,     1,     5,    44,    45,    46,    47,    48,    49,   579,
      43,    43,    40,     5,   749,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,     4,    43,    57,
      43,     5,   767,     5,    43,   605,   767,    43,   608,   609,
     610,     5,     1,    43,     5,   615,   616,   617,     5,     5,
     565,   566,   567,   568,   569,   570,   571,   572,   573,   574,
       5,     1,     5,    40,     5,     3,     5,    44,    45,    46,
      47,    48,    49,     5,     5,   590,   591,   592,   593,   594,
     595,    40,     5,     5,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,   611,   612,   613,   614,
      40,     5,     5,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    40,     5,    57,    43,    44,
      45,    46,    47,    48,    49,   552,   553,   554,   555,   556,
     557,   558,   559,   560,   561,   562,    43,     5,     1,     5,
       5,     5,    41,    50,    51,    52,    53,    54,   575,   576,
      57,     5,     5,   580,   581,   582,   583,   584,   585,   586,
     587,   588,   589,     1,     5,     5,     5,    41,     5,   596,
     597,   598,     5,     5,     5,     5,     5,    40,     1,   606,
     607,    44,    45,    46,    47,    48,    49,     5,     5,     5,
       5,   618,   619,   620,   621,   622,   623,   624,   625,     5,
       5,     5,    40,   137,     5,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    40,     1,    57,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,     5,     1,    57,     5,   751,   752,    40,     5,
       5,    43,   757,   758,     5,     5,   761,   762,    50,    51,
      52,    53,    54,     5,     5,    57,     5,    40,     5,     5,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    40,     5,    57,     5,    44,    45,    46,    47,
      48,    49,     5,     0,     1,     5,     3,     5,    41,     5,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
       5,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,     5,     5,    42,   753,   754,   755,   756,
       5,     5,   759,   760,     5,     5,   763,   764,    55,    56,
       5,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,     1,    40,     5,     5,    43,    44,    45,    46,
      47,    48,    49,     5,    51,    52,    40,     1,     5,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    40,     1,    57,    43,    44,    45,    46,    47,    48,
      49,    40,     5,     5,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    40,     1,    57,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    40,     1,    57,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    40,     1,    57,    43,
      44,    45,    46,    47,    48,    49,    40,     4,    41,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    40,     1,    57,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    40,     1,    57,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    40,     1,    57,     4,    44,    45,    46,    47,    48,
      49,    40,    43,     4,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    40,     1,    57,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    40,     1,    57,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,     4,     1,    57,     4,
      40,    40,     4,     4,    41,    43,    40,     5,     5,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    40,     1,    57,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    40,     1,    57,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    41,     1,    57,     5,     5,    43,     5,     5,     5,
       5,    40,     5,     5,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    40,     1,    57,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    40,     1,    57,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,     5,     1,    57,     5,
       5,    43,     5,    43,     5,     5,    40,     5,     5,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    40,     1,    57,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    40,     1,    57,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,     5,     1,    57,     5,     5,    43,     5,     5,     4,
       4,    40,     5,   749,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    40,     1,    57,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    40,     1,    57,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,     1,   143,   769,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    40,     1,    -1,    -1,
      44,    45,    46,    47,    48,    49,    -1,     1,    -1,    -1,
      -1,    40,    -1,    -1,    -1,    44,    45,    46,    47,    48,
      49,     1,    -1,    -1,    -1,    40,    -1,    -1,    -1,    44,
      45,    46,    47,    48,    49,     1,    40,    -1,    -1,    -1,
      44,    45,    46,    47,    48,    49,    40,     1,    -1,    -1,
      44,    45,    46,    47,    48,    49,    -1,     1,    -1,    -1,
      40,    -1,    -1,    -1,    44,    45,    46,    47,    48,    49,
       1,    -1,    -1,    -1,    40,    -1,    -1,    -1,    44,    45,
      46,    47,    48,    49,     1,    -1,    40,    -1,    -1,    -1,
      44,    45,    46,    47,    48,    49,    40,     1,    -1,    -1,
      44,    45,    46,    47,    48,    49,    -1,     1,    -1,    40,
      -1,    -1,    -1,    44,    45,    46,    47,    48,    49,     1,
      -1,    -1,    -1,    40,    -1,    -1,    -1,    44,    45,    46,
      47,    48,    49,     1,    -1,    -1,    40,    -1,    -1,    -1,
      44,    45,    46,    47,    48,    49,    40,     1,    -1,    -1,
      44,    45,    46,    47,    48,    49,    -1,     1,    40,    -1,
      -1,    -1,    44,    45,    46,    47,    48,    49,     1,    -1,
      -1,    -1,    40,    -1,    -1,    -1,    44,    45,    46,    47,
      48,    49,     1,    -1,    -1,    -1,    40,    -1,    -1,    -1,
      44,    45,    46,    47,    48,    49,    40,     1,    -1,    -1,
      44,    45,    46,    47,    48,    49,    -1,    40,     1,    -1,
      -1,    44,    45,    46,    47,    48,    49,    -1,     1,    -1,
      -1,    40,    -1,    -1,    -1,    44,    45,    46,    47,    48,
      49,     1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    -1,
      44,    45,    46,    47,    48,    49,     1,    40,    -1,    -1,
      -1,    44,    45,    46,    47,    48,    49,    40,     1,    -1,
      -1,    44,    45,    46,    47,    48,    49,    -1,     1,    -1,
      40,    -1,    -1,    -1,    44,    45,    46,    47,    48,    49,
       1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    -1,    44,
      45,    46,    47,    48,    49,     1,    -1,    40,    -1,    -1,
      -1,    44,    45,    46,    47,    48,    49,    40,     1,    -1,
      -1,    44,    45,    46,    47,    48,    49,    -1,     1,    40,
      -1,    -1,    -1,    44,    45,    46,    47,    48,    49,     1,
      -1,    -1,    -1,    -1,    40,    -1,    -1,    -1,    44,    45,
      46,    47,    48,    49,     1,    -1,    -1,    40,    -1,    -1,
      -1,    44,    45,    46,    47,    48,    49,    40,     1,    -1,
      -1,    44,    45,    46,    47,    48,    49,    -1,    40,     1,
      -1,    -1,    44,    45,    46,    47,    48,    49,    -1,     1,
      -1,    -1,    -1,    40,    -1,    -1,    -1,    44,    45,    46,
      47,    48,    49,    -1,    -1,    -1,    -1,    40,    -1,    -1,
      -1,    44,    45,    46,    47,    48,    49,    -1,    40,    -1,
      -1,    -1,    44,    45,    46,    47,    48,    49,    40,    -1,
      -1,    -1,    44,    45,    46,    47,    48,    49,    40,    -1,
      -1,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,   565,   566,   567,   568,   569,   570,   571,
     572,   573,   574
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   160,     0,     1,     3,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    42,    55,    56,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
     148,   149,   150,   151,   152,   153,   154,   155,   156,   157,
     158,   161,   170,   174,   178,   179,   180,   181,   182,   183,
     184,   185,   186,   187,   188,   189,   190,   191,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   202,   203,
     204,   205,   206,   207,   208,   209,   210,   211,   212,   213,
     214,   215,   216,   217,   218,   219,   220,   221,   222,   223,
     224,   225,   226,   227,   228,   229,   230,   231,   232,   233,
     234,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     244,   245,   246,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,     3,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
      40,     4,     4,     4,     4,    40,    44,    45,    46,    47,
      48,    49,   164,   165,   164,    40,    43,    51,    52,    57,
      43,    51,    51,    51,    40,   172,    40,   173,    40,    40,
       6,     1,   164,     1,   164,     1,    43,    50,    51,    52,
      53,    54,    57,   162,   163,   164,   167,   168,     1,   162,
       1,   162,     1,   162,     1,   162,     1,   162,     1,   164,
       1,   164,     1,   164,     1,   164,     1,   164,     1,   164,
       1,   164,     1,   164,     1,   164,     1,   162,     1,   162,
       1,     1,   162,     1,     1,    43,     1,   162,     1,   162,
       1,   164,     1,   164,     1,   164,     1,   164,     1,   164,
       1,   164,     1,   164,     1,   164,     1,   164,     1,   164,
       1,   164,     1,     4,    43,     1,     4,    43,     1,     4,
      43,     1,    43,     1,   164,     1,    43,     1,    43,     1,
     163,     1,   163,     1,     4,    43,     1,     4,    43,     1,
       4,    43,     1,   162,     1,   162,     1,   162,     1,   162,
       1,   162,     1,   162,     1,   162,     1,   164,     1,    43,
       1,    43,     1,    43,     1,    43,     1,    43,     1,     1,
       4,    43,     1,     4,    43,     1,    43,     1,   162,     1,
     164,     1,     4,    43,     1,    40,    43,     1,    40,    43,
       1,    40,    43,     1,    40,    43,     1,   164,     1,   162,
       1,   162,     1,   164,     1,   164,     1,   164,     1,   164,
       1,   164,     1,   164,     1,   164,     1,   164,     1,   162,
       1,   162,     1,   162,     1,     1,   164,     1,   164,     1,
     164,     1,   162,     1,   162,     1,   162,     1,   162,     1,
     162,     1,   162,     1,   162,     1,    51,    52,   164,   169,
       1,   164,   169,     1,   162,     1,   162,     1,   164,     3,
       3,   178,     3,    43,   177,    57,   166,   167,   166,   177,
     177,   177,    41,   177,    40,   166,    40,   177,    41,   177,
       5,     5,     5,     5,     6,     5,     5,     5,     5,     5,
     171,     4,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     3,    41,   177,    41,
      41,    41,    41,     4,    43,     4,    43,     4,     4,    43,
       4,     4,     4,    40,    40,    40,   166,   175,   176,   177,
     162,   162,   162,   162,   162,   162,   162,   162,   162,   162,
     162,   163,     4,    43,   164,   168,   164,   168,   164,   168,
     164,   168,   164,   168,   164,   168,   164,   168,   164,   168,
     164,   168,   164,   168,   162,   162,     4,    43,   163,   162,
     162,   162,   162,   162,   162,   162,   162,   162,   162,   164,
     164,     4,   164,     4,   164,   164,   164,   162,   162,   162,
      40,    43,    40,    43,    40,    43,    40,    43,    40,    43,
      40,    43,   163,   162,   162,   163,   163,   163,    43,   164,
      43,   164,    43,   164,    43,   164,   163,   163,   163,   162,
     162,   162,   162,   162,   162,   162,   162,    41,     5,     5,
      41,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       4,   176,   164,   164,   162,   162,   162,   162,   164,   164,
     162,   162,   164,   164,   162,   162,    43,    43,   166,    43,
     247,   247,     5,     5,     4,    43,     5,     4
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
#line 240 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax, LINE - 1 ); }
    break;

  case 33:
#line 254 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addEntry(); }
    break;

  case 34:
#line 255 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->setModuleName( (yyvsp[(2) - (2)]) ); }
    break;

  case 35:
#line 256 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 36:
#line 257 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 37:
#line 258 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 38:
#line 259 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 39:
#line 260 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 40:
#line 261 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 41:
#line 262 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 42:
#line 263 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 43:
#line 264 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLocal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 44:
#line 265 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addParam( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 45:
#line 266 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 46:
#line 267 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 47:
#line 268 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 48:
#line 269 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 49:
#line 270 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); }
    break;

  case 50:
#line 271 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]) ); }
    break;

  case 51:
#line 272 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 52:
#line 273 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 53:
#line 274 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 54:
#line 275 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); }
    break;

  case 55:
#line 276 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 56:
#line 277 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 57:
#line 278 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 58:
#line 279 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 59:
#line 280 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 60:
#line 281 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (6)]), (yyvsp[(6) - (6)]), (yyvsp[(4) - (6)]) ); }
    break;

  case 61:
#line 282 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDEndSwitch(); }
    break;

  case 62:
#line 283 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 63:
#line 284 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 64:
#line 285 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addPropRef( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 67:
#line 288 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 68:
#line 289 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); }
    break;

  case 69:
#line 290 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 70:
#line 291 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); }
    break;

  case 71:
#line 292 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (2)]) ); }
    break;

  case 72:
#line 293 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (3)]), true ); }
    break;

  case 73:
#line 294 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassCtor( (yyvsp[(2) - (2)]) ); }
    break;

  case 74:
#line 295 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); /* Currently the same as .endfunc */ }
    break;

  case 75:
#line 296 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInherit((yyvsp[(2) - (2)])); }
    break;

  case 77:
#line 297 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFrom( (yyvsp[(2) - (2)]) ); }
    break;

  case 78:
#line 298 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addExtern( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); }
    break;

  case 79:
#line 299 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addDLine( (yyvsp[(2) - (2)]) ); }
    break;

  case 80:
#line 301 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 81:
#line 306 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         (yyvsp[(2) - (2)])->asString().exported( true );
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 82:
#line 312 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      }
    break;

  case 83:
#line 320 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(1) - (1)]) ); }
    break;

  case 84:
#line 321 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(3) - (3)]) ); }
    break;

  case 85:
#line 325 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(1) - (1)]) ); }
    break;

  case 86:
#line 326 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(3) - (3)]) ); }
    break;

  case 87:
#line 329 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->defineLabel( (yyvsp[(1) - (2)])->asLabel() ); }
    break;

  case 89:
#line 333 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(1) - (1)]) ); }
    break;

  case 90:
#line 334 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(3) - (3)]) ); }
    break;

  case 93:
#line 343 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {(yyval) = new Falcon::Pseudo( LINE, (Falcon::int64) 0 ); }
    break;

  case 197:
#line 453 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 198:
#line 454 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LD" ); }
    break;

  case 199:
#line 458 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDRF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 200:
#line 459 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDRF" ); }
    break;

  case 201:
#line 463 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LNIL, (yyvsp[(2) - (2)]) ); }
    break;

  case 202:
#line 464 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LNIL" ); }
    break;

  case 203:
#line 468 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 204:
#line 469 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADD" ); }
    break;

  case 205:
#line 473 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 206:
#line 474 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADDS" ); }
    break;

  case 207:
#line 479 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 208:
#line 480 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUB" ); }
    break;

  case 209:
#line 484 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUBS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 210:
#line 485 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUBS" ); }
    break;

  case 211:
#line 489 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MUL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 212:
#line 490 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MUL" ); }
    break;

  case 213:
#line 494 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MULS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 214:
#line 495 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MULS" ); }
    break;

  case 215:
#line 500 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 216:
#line 501 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIV" ); }
    break;

  case 217:
#line 505 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 218:
#line 506 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIVS" ); }
    break;

  case 219:
#line 510 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MOD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 220:
#line 511 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MOD" ); }
    break;

  case 221:
#line 515 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POW, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 222:
#line 516 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POW" ); }
    break;

  case 223:
#line 521 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 224:
#line 522 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EQ" ); }
    break;

  case 225:
#line 526 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 226:
#line 527 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEQ" ); }
    break;

  case 227:
#line 531 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 228:
#line 532 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GE" ); }
    break;

  case 229:
#line 536 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 230:
#line 537 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GT" ); }
    break;

  case 231:
#line 541 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 232:
#line 542 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LE" ); }
    break;

  case 233:
#line 546 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 234:
#line 547 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LT" ); }
    break;

  case 235:
#line 551 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 236:
#line 552 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); }
    break;

  case 237:
#line 553 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRY" ); }
    break;

  case 238:
#line 557 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INC, (yyvsp[(2) - (2)]) ); }
    break;

  case 239:
#line 558 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INC" ); }
    break;

  case 240:
#line 562 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DEC, (yyvsp[(2) - (2)])  ); }
    break;

  case 241:
#line 563 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DEC" ); }
    break;

  case 242:
#line 568 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INCP, (yyvsp[(2) - (2)]) ); }
    break;

  case 243:
#line 569 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INCP" ); }
    break;

  case 244:
#line 573 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DECP, (yyvsp[(2) - (2)])  ); }
    break;

  case 245:
#line 574 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DECP" ); }
    break;

  case 246:
#line 579 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEG, (yyvsp[(2) - (2)])  ); }
    break;

  case 247:
#line 580 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEG" ); }
    break;

  case 248:
#line 584 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOT, (yyvsp[(2) - (2)])  ); }
    break;

  case 249:
#line 585 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOT" ); }
    break;

  case 250:
#line 589 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 251:
#line 590 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 252:
#line 591 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "CALL" ); }
    break;

  case 253:
#line 595 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 254:
#line 596 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 255:
#line 597 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INST" ); }
    break;

  case 256:
#line 601 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_UNPK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 257:
#line 602 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPK" ); }
    break;

  case 258:
#line 606 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_UNPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 259:
#line 607 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPS" ); }
    break;

  case 260:
#line 612 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PUSH, (yyvsp[(2) - (2)]) ); }
    break;

  case 261:
#line 613 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHN ); }
    break;

  case 262:
#line 614 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PUSH" ); }
    break;

  case 263:
#line 618 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHR, (yyvsp[(2) - (2)]) ); }
    break;

  case 264:
#line 619 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSHR" ); }
    break;

  case 265:
#line 624 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_POP, (yyvsp[(2) - (2)]) ); }
    break;

  case 266:
#line 625 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POP" ); }
    break;

  case 267:
#line 629 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PEEK, (yyvsp[(2) - (2)]) ); }
    break;

  case 268:
#line 630 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PEEK" ); }
    break;

  case 269:
#line 634 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 270:
#line 635 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XPOP" ); }
    break;

  case 271:
#line 640 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 272:
#line 641 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 273:
#line 642 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDV" ); }
    break;

  case 274:
#line 646 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 275:
#line 647 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 276:
#line 648 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVT" ); }
    break;

  case 277:
#line 652 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 278:
#line 653 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 279:
#line 654 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STV" ); }
    break;

  case 280:
#line 658 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 281:
#line 659 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 282:
#line 660 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVR" ); }
    break;

  case 283:
#line 664 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 284:
#line 665 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 285:
#line 666 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVS" ); }
    break;

  case 286:
#line 670 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 287:
#line 671 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 288:
#line 672 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDP" ); yyerrok; }
    break;

  case 289:
#line 676 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 290:
#line 677 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 291:
#line 678 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPT" ); yyerrok; }
    break;

  case 292:
#line 682 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 293:
#line 683 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 294:
#line 684 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STP" ); }
    break;

  case 295:
#line 688 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 296:
#line 689 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 297:
#line 690 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPR" ); }
    break;

  case 298:
#line 694 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 299:
#line 695 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 300:
#line 696 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPS" ); }
    break;

  case 301:
#line 700 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 302:
#line 701 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 303:
#line 702 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAV" ); }
    break;

  case 304:
#line 706 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 305:
#line 707 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 306:
#line 708 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAN" ); }
    break;

  case 307:
#line 712 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 308:
#line 713 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); }
    break;

  case 309:
#line 714 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAL" ); }
    break;

  case 310:
#line 718 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_IPOP, (yyvsp[(2) - (2)]) ); }
    break;

  case 311:
#line 719 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IPOP" ); }
    break;

  case 312:
#line 723 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GENA, (yyvsp[(2) - (2)]) ); }
    break;

  case 313:
#line 724 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENA" ); }
    break;

  case 314:
#line 728 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GEND, (yyvsp[(2) - (2)]) ); }
    break;

  case 315:
#line 729 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEND" ); }
    break;

  case 316:
#line 733 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GENR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); }
    break;

  case 317:
#line 734 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENR" ); }
    break;

  case 318:
#line 738 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GEOR, (yyvsp[(2) - (2)]) ); }
    break;

  case 319:
#line 739 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEOR" ); }
    break;

  case 320:
#line 743 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RIS, (yyvsp[(2) - (2)]) ); }
    break;

  case 321:
#line 744 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RIS" ); }
    break;

  case 322:
#line 748 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 323:
#line 749 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); }
    break;

  case 324:
#line 750 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JMP" ); }
    break;

  case 325:
#line 754 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOOL, (yyvsp[(1) - (2)]) ); }
    break;

  case 326:
#line 755 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOOL" ); }
    break;

  case 327:
#line 759 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 328:
#line 760 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 329:
#line 761 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFT" ); }
    break;

  case 330:
#line 765 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 331:
#line 766 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 332:
#line 767 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFF" ); }
    break;

  case 333:
#line 772 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 334:
#line 773 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 335:
#line 774 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORK" ); }
    break;

  case 336:
#line 778 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 337:
#line 779 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 338:
#line 780 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JTRY" ); }
    break;

  case 339:
#line 784 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RET ); }
    break;

  case 340:
#line 785 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RET" ); }
    break;

  case 341:
#line 789 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETA ); }
    break;

  case 342:
#line 790 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETA" ); }
    break;

  case 343:
#line 794 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETV, (yyvsp[(2) - (2)]) ); }
    break;

  case 344:
#line 795 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETV" ); }
    break;

  case 345:
#line 799 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOP ); }
    break;

  case 346:
#line 800 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOP" ); }
    break;

  case 347:
#line 804 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_PTRY, (yyvsp[(2) - (2)]) ); }
    break;

  case 348:
#line 805 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PTRY" ); }
    break;

  case 349:
#line 809 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_END ); }
    break;

  case 350:
#line 810 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "END" ); }
    break;

  case 351:
#line 814 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 352:
#line 815 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SWCH" ); }
    break;

  case 353:
#line 819 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); }
    break;

  case 354:
#line 820 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SELE" ); }
    break;

  case 355:
#line 825 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         Falcon::Pseudo *psd = new Falcon::Pseudo( Falcon::Pseudo::tswitch_list );
         psd->line( LINE );
         psd->asList()->pushBack( (yyvsp[(1) - (3)]) );
         psd->asList()->pushBack( (yyvsp[(3) - (3)]) );
         (yyval) = psd;
      }
    break;

  case 356:
#line 834 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(3) - (5)]) );
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(5) - (5)]) );
         (yyval) = (yyvsp[(1) - (5)]);
      }
    break;

  case 357:
#line 842 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 358:
#line 843 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); }
    break;

  case 359:
#line 844 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ONCE" ); }
    break;

  case 360:
#line 848 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 361:
#line 849 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 362:
#line 850 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 363:
#line 851 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 364:
#line 852 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BAND" ); }
    break;

  case 365:
#line 856 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 366:
#line 857 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 367:
#line 858 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 368:
#line 859 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 369:
#line 860 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOR" ); }
    break;

  case 370:
#line 864 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 371:
#line 865 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 372:
#line 866 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 373:
#line 867 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 374:
#line 868 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 375:
#line 872 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 376:
#line 873 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); }
    break;

  case 377:
#line 874 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); }
    break;

  case 378:
#line 878 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_AND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 379:
#line 879 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "AND" ); }
    break;

  case 380:
#line 883 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_OR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 381:
#line 884 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "OR" ); }
    break;

  case 382:
#line 888 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ANDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 383:
#line 889 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ANDS" ); }
    break;

  case 384:
#line 893 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 385:
#line 894 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ORS" ); }
    break;

  case 386:
#line 898 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 387:
#line 899 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XORS" ); }
    break;

  case 388:
#line 903 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MODS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 389:
#line 904 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MODS" ); }
    break;

  case 390:
#line 908 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POWS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 391:
#line 909 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POWS" ); }
    break;

  case 392:
#line 913 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOTS, (yyvsp[(2) - (2)]) ); }
    break;

  case 393:
#line 914 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOTS" ); }
    break;

  case 394:
#line 918 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 395:
#line 919 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 396:
#line 920 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HAS" ); }
    break;

  case 397:
#line 924 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 398:
#line 925 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 399:
#line 926 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HASN" ); }
    break;

  case 400:
#line 930 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 401:
#line 931 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 402:
#line 932 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVE" ); }
    break;

  case 403:
#line 936 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 404:
#line 937 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 405:
#line 938 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVN" ); }
    break;

  case 406:
#line 943 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_IN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 407:
#line 944 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IN" ); }
    break;

  case 408:
#line 948 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOIN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 409:
#line 949 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOIN" ); }
    break;

  case 410:
#line 953 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PROV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); }
    break;

  case 411:
#line 954 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PROV" ); }
    break;

  case 412:
#line 958 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSIN, (yyvsp[(2) - (2)]) ); }
    break;

  case 413:
#line 959 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSIN" ); }
    break;

  case 414:
#line 963 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PASS, (yyvsp[(2) - (2)]) ); }
    break;

  case 415:
#line 964 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PASS" ); }
    break;

  case 416:
#line 968 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 417:
#line 969 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHR" ); }
    break;

  case 418:
#line 973 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 419:
#line 974 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHL" ); }
    break;

  case 420:
#line 978 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHRS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 421:
#line 979 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHRS" ); }
    break;

  case 422:
#line 983 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHLS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 423:
#line 984 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHLS" ); }
    break;

  case 424:
#line 988 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 425:
#line 989 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVR" ); }
    break;

  case 426:
#line 993 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 427:
#line 994 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPR" ); }
    break;

  case 428:
#line 998 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LSB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 429:
#line 999 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LSB" ); }
    break;

  case 430:
#line 1003 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 431:
#line 1004 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); }
    break;

  case 432:
#line 1005 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INDI" ); }
    break;

  case 433:
#line 1009 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 434:
#line 1010 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); }
    break;

  case 435:
#line 1011 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "STEX" ); }
    break;

  case 436:
#line 1015 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_TRAC, (yyvsp[(2) - (2)]) ); }
    break;

  case 437:
#line 1016 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "TRAC" ); }
    break;

  case 438:
#line 1020 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_WRT, (yyvsp[(2) - (2)]) ); }
    break;

  case 439:
#line 1021 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "WRT" ); }
    break;

  case 440:
#line 1026 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STO, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); }
    break;

  case 441:
#line 1027 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STO" ); }
    break;


/* Line 1267 of yacc.c.  */
#line 4194 "/home/user/Progetti/falcon/core/engine/fasm_parser.cpp"
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


#line 1031 "/home/user/Progetti/falcon/core/engine/fasm_parser.yy"
 /* c code */


/****************************************************
* C Code for falcon HSM compiler
*****************************************************/


void fasm_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of falcon_parser.yxx */


