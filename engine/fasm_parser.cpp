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
     DCSTRING = 289,
     DHAS = 290,
     DHASNT = 291,
     DINHERIT = 292,
     DINSTANCE = 293,
     SYMBOL = 294,
     EXPORT = 295,
     LABEL = 296,
     INTEGER = 297,
     REG_A = 298,
     REG_B = 299,
     REG_S1 = 300,
     REG_S2 = 301,
     NUMERIC = 302,
     STRING = 303,
     STRING_ID = 304,
     I_LD = 305,
     I_LNIL = 306,
     NIL = 307,
     I_ADD = 308,
     I_SUB = 309,
     I_MUL = 310,
     I_DIV = 311,
     I_MOD = 312,
     I_POW = 313,
     I_ADDS = 314,
     I_SUBS = 315,
     I_MULS = 316,
     I_DIVS = 317,
     I_POWS = 318,
     I_INC = 319,
     I_DEC = 320,
     I_NEG = 321,
     I_NOT = 322,
     I_RET = 323,
     I_RETV = 324,
     I_RETA = 325,
     I_FORK = 326,
     I_PUSH = 327,
     I_PSHR = 328,
     I_PSHN = 329,
     I_POP = 330,
     I_LDV = 331,
     I_LDVT = 332,
     I_STV = 333,
     I_STVR = 334,
     I_STVS = 335,
     I_LDP = 336,
     I_LDPT = 337,
     I_STP = 338,
     I_STPR = 339,
     I_STPS = 340,
     I_TRAV = 341,
     I_TRAN = 342,
     I_TRAL = 343,
     I_IPOP = 344,
     I_XPOP = 345,
     I_GENA = 346,
     I_GEND = 347,
     I_GENR = 348,
     I_GEOR = 349,
     I_JMP = 350,
     I_IFT = 351,
     I_IFF = 352,
     I_BOOL = 353,
     I_EQ = 354,
     I_NEQ = 355,
     I_GT = 356,
     I_GE = 357,
     I_LT = 358,
     I_LE = 359,
     I_UNPK = 360,
     I_UNPS = 361,
     I_CALL = 362,
     I_INST = 363,
     I_SWCH = 364,
     I_SELE = 365,
     I_NOP = 366,
     I_TRY = 367,
     I_JTRY = 368,
     I_PTRY = 369,
     I_RIS = 370,
     I_LDRF = 371,
     I_ONCE = 372,
     I_BAND = 373,
     I_BOR = 374,
     I_BXOR = 375,
     I_BNOT = 376,
     I_MODS = 377,
     I_AND = 378,
     I_OR = 379,
     I_ANDS = 380,
     I_ORS = 381,
     I_XORS = 382,
     I_NOTS = 383,
     I_HAS = 384,
     I_HASN = 385,
     I_GIVE = 386,
     I_GIVN = 387,
     I_IN = 388,
     I_NOIN = 389,
     I_PROV = 390,
     I_END = 391,
     I_PEEK = 392,
     I_PSIN = 393,
     I_PASS = 394,
     I_FORI = 395,
     I_FORN = 396,
     I_SHR = 397,
     I_SHL = 398,
     I_SHRS = 399,
     I_SHLS = 400,
     I_LDVR = 401,
     I_LDPR = 402,
     I_LSB = 403,
     I_INDI = 404,
     I_STEX = 405,
     I_TRAC = 406,
     I_WRT = 407
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
#define DCSTRING 289
#define DHAS 290
#define DHASNT 291
#define DINHERIT 292
#define DINSTANCE 293
#define SYMBOL 294
#define EXPORT 295
#define LABEL 296
#define INTEGER 297
#define REG_A 298
#define REG_B 299
#define REG_S1 300
#define REG_S2 301
#define NUMERIC 302
#define STRING 303
#define STRING_ID 304
#define I_LD 305
#define I_LNIL 306
#define NIL 307
#define I_ADD 308
#define I_SUB 309
#define I_MUL 310
#define I_DIV 311
#define I_MOD 312
#define I_POW 313
#define I_ADDS 314
#define I_SUBS 315
#define I_MULS 316
#define I_DIVS 317
#define I_POWS 318
#define I_INC 319
#define I_DEC 320
#define I_NEG 321
#define I_NOT 322
#define I_RET 323
#define I_RETV 324
#define I_RETA 325
#define I_FORK 326
#define I_PUSH 327
#define I_PSHR 328
#define I_PSHN 329
#define I_POP 330
#define I_LDV 331
#define I_LDVT 332
#define I_STV 333
#define I_STVR 334
#define I_STVS 335
#define I_LDP 336
#define I_LDPT 337
#define I_STP 338
#define I_STPR 339
#define I_STPS 340
#define I_TRAV 341
#define I_TRAN 342
#define I_TRAL 343
#define I_IPOP 344
#define I_XPOP 345
#define I_GENA 346
#define I_GEND 347
#define I_GENR 348
#define I_GEOR 349
#define I_JMP 350
#define I_IFT 351
#define I_IFF 352
#define I_BOOL 353
#define I_EQ 354
#define I_NEQ 355
#define I_GT 356
#define I_GE 357
#define I_LT 358
#define I_LE 359
#define I_UNPK 360
#define I_UNPS 361
#define I_CALL 362
#define I_INST 363
#define I_SWCH 364
#define I_SELE 365
#define I_NOP 366
#define I_TRY 367
#define I_JTRY 368
#define I_PTRY 369
#define I_RIS 370
#define I_LDRF 371
#define I_ONCE 372
#define I_BAND 373
#define I_BOR 374
#define I_BXOR 375
#define I_BNOT 376
#define I_MODS 377
#define I_AND 378
#define I_OR 379
#define I_ANDS 380
#define I_ORS 381
#define I_XORS 382
#define I_NOTS 383
#define I_HAS 384
#define I_HASN 385
#define I_GIVE 386
#define I_GIVN 387
#define I_IN 388
#define I_NOIN 389
#define I_PROV 390
#define I_END 391
#define I_PEEK 392
#define I_PSIN 393
#define I_PASS 394
#define I_FORI 395
#define I_FORN 396
#define I_SHR 397
#define I_SHL 398
#define I_SHRS 399
#define I_SHLS 400
#define I_LDVR 401
#define I_LDPR 402
#define I_LSB 403
#define I_INDI 404
#define I_STEX 405
#define I_TRAC 406
#define I_WRT 407




/* Copy the first part of user declarations.  */
#line 23 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"

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
#line 456 "/home/gian/Progetti/falcon/core/engine/fasm_parser.cpp"

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
#define YYLAST   1473

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  153
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  122
/* YYNRULES -- Number of rules.  */
#define YYNRULES  435
/* YYNRULES -- Number of states.  */
#define YYNSTATES  798

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   407

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
     145,   146,   147,   148,   149,   150,   151,   152
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,     9,    12,    16,    19,    22,
      25,    27,    29,    31,    33,    35,    37,    39,    41,    43,
      45,    47,    49,    51,    53,    55,    57,    59,    61,    63,
      65,    68,    72,    77,    82,    88,    92,    97,   101,   106,
     110,   114,   117,   121,   125,   130,   132,   135,   140,   145,
     150,   155,   160,   165,   170,   175,   180,   187,   189,   193,
     197,   201,   204,   207,   212,   218,   222,   227,   230,   234,
     237,   239,   240,   245,   248,   252,   255,   258,   261,   263,
     267,   269,   273,   276,   277,   279,   283,   285,   287,   288,
     290,   292,   294,   296,   298,   300,   302,   304,   306,   308,
     310,   312,   314,   316,   318,   320,   322,   324,   326,   328,
     330,   332,   334,   336,   338,   340,   342,   344,   346,   348,
     350,   352,   354,   356,   358,   360,   362,   364,   366,   368,
     370,   372,   374,   376,   378,   380,   382,   384,   386,   388,
     390,   392,   394,   396,   398,   400,   402,   404,   406,   408,
     410,   412,   414,   416,   418,   420,   422,   424,   426,   428,
     430,   432,   434,   436,   438,   440,   442,   444,   446,   448,
     450,   452,   454,   456,   458,   460,   462,   464,   466,   468,
     470,   472,   474,   476,   478,   480,   482,   484,   486,   488,
     490,   492,   497,   500,   505,   508,   511,   514,   519,   522,
     527,   530,   535,   538,   543,   546,   551,   554,   559,   562,
     567,   570,   575,   578,   583,   586,   591,   594,   599,   602,
     607,   610,   615,   618,   623,   626,   631,   634,   639,   642,
     645,   648,   651,   654,   657,   660,   663,   666,   669,   672,
     675,   680,   685,   688,   693,   698,   701,   706,   709,   714,
     717,   720,   722,   725,   728,   731,   734,   737,   740,   743,
     746,   749,   754,   759,   762,   769,   776,   779,   786,   793,
     796,   803,   810,   813,   818,   823,   826,   831,   836,   839,
     846,   853,   856,   863,   870,   873,   880,   887,   890,   895,
     900,   903,   910,   917,   920,   927,   934,   937,   940,   943,
     946,   949,   952,   955,   958,   961,   964,   969,   972,   975,
     978,   981,   984,   987,   990,   993,   996,   999,  1004,  1009,
    1012,  1017,  1022,  1025,  1030,  1035,  1038,  1041,  1044,  1047,
    1049,  1052,  1054,  1057,  1060,  1063,  1065,  1068,  1071,  1074,
    1076,  1079,  1086,  1089,  1096,  1099,  1103,  1109,  1114,  1119,
    1122,  1127,  1132,  1137,  1142,  1145,  1150,  1155,  1160,  1165,
    1168,  1173,  1178,  1183,  1188,  1191,  1194,  1197,  1200,  1205,
    1208,  1213,  1216,  1221,  1224,  1229,  1232,  1237,  1240,  1245,
    1248,  1253,  1256,  1259,  1262,  1267,  1272,  1275,  1280,  1285,
    1288,  1293,  1298,  1301,  1306,  1311,  1314,  1319,  1322,  1327,
    1330,  1335,  1338,  1341,  1344,  1347,  1350,  1357,  1364,  1367,
    1372,  1377,  1380,  1385,  1388,  1393,  1396,  1401,  1404,  1409,
    1412,  1417,  1420,  1425,  1428,  1433,  1436,  1439,  1442,  1445,
    1448,  1451,  1454,  1457,  1460,  1463
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     154,     0,    -1,    -1,   154,   155,    -1,     3,    -1,   164,
       3,    -1,   168,   172,     3,    -1,   168,     3,    -1,   172,
       3,    -1,     1,     3,    -1,    52,    -1,   157,    -1,   158,
      -1,   161,    -1,    39,    -1,   159,    -1,    43,    -1,    44,
      -1,    45,    -1,    46,    -1,    52,    -1,   161,    -1,    47,
      -1,   162,    -1,    49,    -1,    48,    -1,    42,    -1,    49,
      -1,    48,    -1,     7,    -1,    26,     4,    -1,     8,     4,
     171,    -1,     8,     4,   171,    40,    -1,     9,     4,   160,
     171,    -1,     9,     4,   160,   171,    40,    -1,    10,     4,
     160,    -1,    10,     4,   160,    40,    -1,    11,     4,   171,
      -1,    11,     4,   171,    40,    -1,    12,     4,   171,    -1,
      13,     4,   171,    -1,    14,     4,    -1,    14,     4,    40,
      -1,    16,     4,   171,    -1,    16,     4,   171,    40,    -1,
      15,    -1,    27,     4,    -1,    28,   158,     5,     4,    -1,
      28,   158,     5,    42,    -1,    29,   158,     5,     4,    -1,
      29,   158,     5,    42,    -1,    30,    52,     5,     4,    -1,
      30,    42,     5,     4,    -1,    30,    48,     5,     4,    -1,
      30,    49,     5,     4,    -1,    30,    39,     5,     4,    -1,
      30,    42,     6,    42,     5,     4,    -1,    31,    -1,    18,
       4,   160,    -1,    18,     4,    39,    -1,    19,     4,    39,
      -1,    35,   166,    -1,    36,   167,    -1,    38,    39,     4,
     171,    -1,    38,    39,     4,   171,    40,    -1,    20,     4,
     171,    -1,    20,     4,   171,    40,    -1,    21,     4,    -1,
      21,     4,    40,    -1,    22,    39,    -1,    23,    -1,    -1,
      37,    39,   165,   169,    -1,    24,     4,    -1,    25,     4,
     171,    -1,    32,    42,    -1,    33,    48,    -1,    34,    48,
      -1,    39,    -1,   166,     5,    39,    -1,    39,    -1,   167,
       5,    39,    -1,    41,     6,    -1,    -1,   170,    -1,   169,
       5,   170,    -1,   160,    -1,    39,    -1,    -1,    42,    -1,
     173,    -1,   175,    -1,   176,    -1,   178,    -1,   180,    -1,
     182,    -1,   184,    -1,   185,    -1,   177,    -1,   179,    -1,
     181,    -1,   183,    -1,   193,    -1,   194,    -1,   186,    -1,
     187,    -1,   188,    -1,   189,    -1,   190,    -1,   191,    -1,
     195,    -1,   196,    -1,   201,    -1,   202,    -1,   203,    -1,
     205,    -1,   206,    -1,   207,    -1,   208,    -1,   209,    -1,
     210,    -1,   211,    -1,   212,    -1,   213,    -1,   215,    -1,
     214,    -1,   216,    -1,   217,    -1,   218,    -1,   219,    -1,
     220,    -1,   221,    -1,   222,    -1,   223,    -1,   225,    -1,
     226,    -1,   227,    -1,   228,    -1,   229,    -1,   199,    -1,
     200,    -1,   197,    -1,   198,    -1,   231,    -1,   233,    -1,
     232,    -1,   234,    -1,   192,    -1,   235,    -1,   230,    -1,
     224,    -1,   237,    -1,   238,    -1,   174,    -1,   236,    -1,
     241,    -1,   242,    -1,   243,    -1,   244,    -1,   245,    -1,
     246,    -1,   247,    -1,   248,    -1,   249,    -1,   252,    -1,
     250,    -1,   251,    -1,   253,    -1,   254,    -1,   255,    -1,
     256,    -1,   257,    -1,   258,    -1,   259,    -1,   240,    -1,
     204,    -1,   260,    -1,   261,    -1,   262,    -1,   263,    -1,
     264,    -1,   265,    -1,   266,    -1,   267,    -1,   268,    -1,
     269,    -1,   270,    -1,   271,    -1,   272,    -1,   273,    -1,
     274,    -1,    50,   158,     5,   156,    -1,    50,     1,    -1,
     116,   158,     5,   156,    -1,   116,     1,    -1,    51,   158,
      -1,    51,     1,    -1,    53,   156,     5,   156,    -1,    53,
       1,    -1,    59,   158,     5,   156,    -1,    59,     1,    -1,
      54,   156,     5,   156,    -1,    54,     1,    -1,    60,   158,
       5,   156,    -1,    60,     1,    -1,    55,   156,     5,   156,
      -1,    55,     1,    -1,    61,   158,     5,   156,    -1,    61,
       1,    -1,    56,   156,     5,   156,    -1,    56,     1,    -1,
      62,   158,     5,   156,    -1,    62,     1,    -1,    57,   156,
       5,   156,    -1,    57,     1,    -1,    58,   156,     5,   156,
      -1,    58,     1,    -1,    99,   156,     5,   156,    -1,    99,
       1,    -1,   100,   156,     5,   156,    -1,   100,     1,    -1,
     102,   156,     5,   156,    -1,   102,     1,    -1,   101,   156,
       5,   156,    -1,   101,     1,    -1,   104,   156,     5,   156,
      -1,   104,     1,    -1,   103,   156,     5,   156,    -1,   103,
       1,    -1,   112,     4,    -1,   112,    42,    -1,   112,     1,
      -1,    64,   158,    -1,    64,     1,    -1,    65,   158,    -1,
      65,     1,    -1,    66,   156,    -1,    66,     1,    -1,    67,
     156,    -1,    67,     1,    -1,   107,    42,     5,   158,    -1,
     107,    42,     5,     4,    -1,   107,     1,    -1,   108,    42,
       5,   158,    -1,   108,    42,     5,     4,    -1,   108,     1,
      -1,   105,   158,     5,   158,    -1,   105,     1,    -1,   106,
      42,     5,   158,    -1,   106,     1,    -1,    72,   156,    -1,
      74,    -1,    72,     1,    -1,    73,   156,    -1,    73,     1,
      -1,    75,   158,    -1,    75,     1,    -1,   137,   158,    -1,
     137,     1,    -1,    90,   158,    -1,    90,     1,    -1,    76,
     158,     5,   158,    -1,    76,   158,     5,   162,    -1,    76,
       1,    -1,    77,   158,     5,   158,     5,   158,    -1,    77,
     158,     5,   162,     5,   158,    -1,    77,     1,    -1,    78,
     158,     5,   158,     5,   156,    -1,    78,   158,     5,   162,
       5,   156,    -1,    78,     1,    -1,    79,   158,     5,   158,
       5,   156,    -1,    79,   158,     5,   162,     5,   156,    -1,
      79,     1,    -1,    80,   158,     5,   158,    -1,    80,   158,
       5,   162,    -1,    80,     1,    -1,    81,   158,     5,   158,
      -1,    81,   158,     5,   162,    -1,    81,     1,    -1,    82,
     158,     5,   158,     5,   158,    -1,    82,   158,     5,   162,
       5,   158,    -1,    82,     1,    -1,    83,   158,     5,   158,
       5,   156,    -1,    83,   158,     5,   162,     5,   156,    -1,
      83,     1,    -1,    84,   158,     5,   158,     5,   158,    -1,
      84,   158,     5,   162,     5,   158,    -1,    84,     1,    -1,
      85,   158,     5,   158,    -1,    85,   158,     5,   162,    -1,
      85,     1,    -1,    86,    42,     5,   156,     5,   156,    -1,
      86,     4,     5,   156,     5,   156,    -1,    86,     1,    -1,
      87,     4,     5,     4,     5,    42,    -1,    87,    42,     5,
      42,     5,    42,    -1,    87,     1,    -1,    88,    42,    -1,
      88,     4,    -1,    88,     1,    -1,    89,    42,    -1,    89,
       1,    -1,    91,    42,    -1,    91,     1,    -1,    92,    42,
      -1,    92,     1,    -1,    93,   157,     5,   157,    -1,    93,
       1,    -1,    94,   157,    -1,    94,     1,    -1,   115,   156,
      -1,   115,     1,    -1,    95,     4,    -1,    95,    42,    -1,
      95,     1,    -1,    98,   156,    -1,    98,     1,    -1,    96,
       4,     5,   156,    -1,    96,    42,     5,   156,    -1,    96,
       1,    -1,    97,     4,     5,   156,    -1,    97,    42,     5,
     156,    -1,    97,     1,    -1,    71,    42,     5,     4,    -1,
      71,    42,     5,    42,    -1,    71,     1,    -1,   113,     4,
      -1,   113,    42,    -1,   113,     1,    -1,    68,    -1,    68,
       1,    -1,    70,    -1,    70,     1,    -1,    69,   156,    -1,
      69,     1,    -1,   111,    -1,   111,     1,    -1,   114,    42,
      -1,   114,     1,    -1,   136,    -1,   136,     1,    -1,   109,
      42,     5,   158,     5,   239,    -1,   109,     1,    -1,   110,
      42,     5,   158,     5,   239,    -1,   110,     1,    -1,    42,
       5,     4,    -1,   239,     5,    42,     5,     4,    -1,   117,
      42,     5,   156,    -1,   117,     4,     5,   156,    -1,   117,
       1,    -1,   118,    39,     5,    39,    -1,   118,    39,     5,
      42,    -1,   118,    42,     5,    39,    -1,   118,    42,     5,
      42,    -1,   118,     1,    -1,   119,    39,     5,    39,    -1,
     119,    39,     5,    42,    -1,   119,    42,     5,    39,    -1,
     119,    42,     5,    42,    -1,   119,     1,    -1,   120,    39,
       5,    39,    -1,   120,    39,     5,    42,    -1,   120,    42,
       5,    39,    -1,   120,    42,     5,    42,    -1,   120,     1,
      -1,   121,    39,    -1,   121,    42,    -1,   121,     1,    -1,
     123,   156,     5,   156,    -1,   123,     1,    -1,   124,   156,
       5,   156,    -1,   124,     1,    -1,   125,   158,     5,   157,
      -1,   125,     1,    -1,   126,   158,     5,   157,    -1,   126,
       1,    -1,   127,   158,     5,   157,    -1,   127,     1,    -1,
     122,   158,     5,   157,    -1,   122,     1,    -1,    63,   158,
       5,   157,    -1,    63,     1,    -1,   128,   158,    -1,   128,
       1,    -1,   129,   158,     5,   158,    -1,   129,   158,     5,
      42,    -1,   129,     1,    -1,   130,   158,     5,   158,    -1,
     130,   158,     5,    42,    -1,   130,     1,    -1,   131,   158,
       5,   158,    -1,   131,   158,     5,    42,    -1,   131,     1,
      -1,   132,   158,     5,   158,    -1,   132,   158,     5,    42,
      -1,   132,     1,    -1,   133,   156,     5,   157,    -1,   133,
       1,    -1,   134,   156,     5,   157,    -1,   134,     1,    -1,
     135,   156,     5,   157,    -1,   135,     1,    -1,   138,   158,
      -1,   138,     1,    -1,   139,   158,    -1,   139,     1,    -1,
     140,     4,     5,    39,     5,   156,    -1,   140,    42,     5,
      39,     5,   156,    -1,   140,     1,    -1,   141,     4,     5,
      39,    -1,   141,    42,     5,    39,    -1,   141,     1,    -1,
     142,   156,     5,   156,    -1,   142,     1,    -1,   143,   156,
       5,   156,    -1,   143,     1,    -1,   144,   156,     5,   156,
      -1,   144,     1,    -1,   145,   156,     5,   156,    -1,   145,
       1,    -1,   146,   156,     5,   156,    -1,   146,     1,    -1,
     147,   156,     5,   156,    -1,   147,     1,    -1,   148,   156,
       5,   156,    -1,   148,     1,    -1,   149,   163,    -1,   149,
     158,    -1,   149,     1,    -1,   150,   163,    -1,   150,   158,
      -1,   150,     1,    -1,   151,   156,    -1,   151,     1,    -1,
     152,   156,    -1,   152,     1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   230,   230,   232,   236,   237,   238,   239,   240,   241,
     244,   244,   245,   245,   246,   246,   247,   247,   247,   247,
     249,   249,   250,   250,   251,   251,   251,   252,   252,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,   294,   295,
     296,   297,   297,   298,   299,   300,   301,   306,   315,   316,
     320,   321,   324,   326,   328,   329,   333,   334,   338,   339,
     343,   344,   345,   346,   347,   348,   349,   350,   351,   352,
     353,   354,   355,   356,   357,   358,   359,   360,   361,   362,
     363,   364,   365,   366,   367,   368,   369,   370,   371,   372,
     373,   374,   375,   376,   377,   378,   379,   380,   381,   382,
     383,   384,   385,   386,   387,   388,   389,   390,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   401,   402,
     403,   404,   405,   406,   407,   408,   409,   410,   411,   412,
     413,   414,   415,   416,   417,   418,   419,   420,   421,   422,
     423,   424,   425,   426,   427,   428,   429,   430,   431,   432,
     433,   434,   435,   436,   437,   438,   439,   440,   441,   442,
     443,   447,   448,   452,   453,   457,   458,   462,   463,   467,
     468,   473,   474,   478,   479,   483,   484,   488,   489,   494,
     495,   499,   500,   504,   505,   509,   510,   515,   516,   520,
     521,   525,   526,   530,   531,   535,   536,   540,   541,   545,
     546,   547,   551,   552,   556,   557,   561,   562,   566,   567,
     571,   572,   573,   577,   578,   579,   583,   584,   588,   589,
     594,   595,   596,   600,   601,   606,   607,   611,   612,   616,
     617,   622,   623,   624,   628,   629,   630,   634,   635,   636,
     640,   641,   642,   646,   647,   648,   652,   653,   654,   658,
     659,   660,   664,   665,   666,   670,   671,   672,   676,   677,
     678,   682,   683,   684,   688,   689,   690,   694,   695,   696,
     700,   701,   705,   706,   710,   711,   715,   716,   720,   721,
     725,   726,   730,   731,   732,   736,   737,   741,   742,   743,
     747,   748,   749,   754,   755,   756,   760,   761,   762,   766,
     767,   771,   772,   776,   777,   781,   782,   786,   787,   791,
     792,   796,   797,   801,   802,   806,   815,   824,   825,   826,
     830,   831,   832,   833,   834,   838,   839,   840,   841,   842,
     846,   847,   848,   849,   850,   854,   855,   856,   860,   861,
     865,   866,   870,   871,   875,   876,   880,   881,   885,   886,
     890,   891,   895,   896,   900,   901,   902,   906,   907,   908,
     912,   913,   914,   918,   919,   920,   925,   926,   930,   931,
     935,   936,   940,   941,   945,   946,   950,   951,   952,   956,
     957,   958,   962,   963,   967,   968,   972,   973,   977,   978,
     982,   983,   987,   988,   992,   993,   997,   998,   999,  1003,
    1004,  1005,  1009,  1010,  1014,  1015
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
  "DSTRING", "DCSTRING", "DHAS", "DHASNT", "DINHERIT", "DINSTANCE",
  "SYMBOL", "EXPORT", "LABEL", "INTEGER", "REG_A", "REG_B", "REG_S1",
  "REG_S2", "NUMERIC", "STRING", "STRING_ID", "I_LD", "I_LNIL", "NIL",
  "I_ADD", "I_SUB", "I_MUL", "I_DIV", "I_MOD", "I_POW", "I_ADDS", "I_SUBS",
  "I_MULS", "I_DIVS", "I_POWS", "I_INC", "I_DEC", "I_NEG", "I_NOT",
  "I_RET", "I_RETV", "I_RETA", "I_FORK", "I_PUSH", "I_PSHR", "I_PSHN",
  "I_POP", "I_LDV", "I_LDVT", "I_STV", "I_STVR", "I_STVS", "I_LDP",
  "I_LDPT", "I_STP", "I_STPR", "I_STPS", "I_TRAV", "I_TRAN", "I_TRAL",
  "I_IPOP", "I_XPOP", "I_GENA", "I_GEND", "I_GENR", "I_GEOR", "I_JMP",
  "I_IFT", "I_IFF", "I_BOOL", "I_EQ", "I_NEQ", "I_GT", "I_GE", "I_LT",
  "I_LE", "I_UNPK", "I_UNPS", "I_CALL", "I_INST", "I_SWCH", "I_SELE",
  "I_NOP", "I_TRY", "I_JTRY", "I_PTRY", "I_RIS", "I_LDRF", "I_ONCE",
  "I_BAND", "I_BOR", "I_BXOR", "I_BNOT", "I_MODS", "I_AND", "I_OR",
  "I_ANDS", "I_ORS", "I_XORS", "I_NOTS", "I_HAS", "I_HASN", "I_GIVE",
  "I_GIVN", "I_IN", "I_NOIN", "I_PROV", "I_END", "I_PEEK", "I_PSIN",
  "I_PASS", "I_FORI", "I_FORN", "I_SHR", "I_SHL", "I_SHRS", "I_SHLS",
  "I_LDVR", "I_LDPR", "I_LSB", "I_INDI", "I_STEX", "I_TRAC", "I_WRT",
  "$accept", "input", "line", "xoperand", "operand", "op_variable",
  "op_register", "x_op_immediate", "op_immediate", "op_scalar",
  "op_string", "directive", "@1", "has_symlist", "hasnt_symlist", "label",
  "inherit_param_list", "inherit_param", "def_line", "instruction",
  "inst_ld", "inst_ldrf", "inst_ldnil", "inst_add", "inst_adds",
  "inst_sub", "inst_subs", "inst_mul", "inst_muls", "inst_div",
  "inst_divs", "inst_mod", "inst_pow", "inst_eq", "inst_ne", "inst_ge",
  "inst_gt", "inst_le", "inst_lt", "inst_try", "inst_inc", "inst_dec",
  "inst_neg", "inst_not", "inst_call", "inst_inst", "inst_unpk",
  "inst_unps", "inst_push", "inst_pshr", "inst_pop", "inst_peek",
  "inst_xpop", "inst_ldv", "inst_ldvt", "inst_stv", "inst_stvr",
  "inst_stvs", "inst_ldp", "inst_ldpt", "inst_stp", "inst_stpr",
  "inst_stps", "inst_trav", "inst_tran", "inst_tral", "inst_ipop",
  "inst_gena", "inst_gend", "inst_genr", "inst_geor", "inst_ris",
  "inst_jmp", "inst_bool", "inst_ift", "inst_iff", "inst_fork",
  "inst_jtry", "inst_ret", "inst_reta", "inst_retval", "inst_nop",
  "inst_ptry", "inst_end", "inst_swch", "inst_sele", "switch_list",
  "inst_once", "inst_band", "inst_bor", "inst_bxor", "inst_bnot",
  "inst_and", "inst_or", "inst_ands", "inst_ors", "inst_xors", "inst_mods",
  "inst_pows", "inst_nots", "inst_has", "inst_hasn", "inst_give",
  "inst_givn", "inst_in", "inst_noin", "inst_prov", "inst_psin",
  "inst_pass", "inst_fori", "inst_forn", "inst_shr", "inst_shl",
  "inst_shrs", "inst_shls", "inst_ldvr", "inst_ldpr", "inst_lsb",
  "inst_indi", "inst_stex", "inst_trac", "inst_wrt", 0
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
     405,   406,   407
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   153,   154,   154,   155,   155,   155,   155,   155,   155,
     156,   156,   157,   157,   158,   158,   159,   159,   159,   159,
     160,   160,   161,   161,   162,   162,   162,   163,   163,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   165,   164,   164,   164,   164,   164,   164,   166,   166,
     167,   167,   168,   169,   169,   169,   170,   170,   171,   171,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   173,   173,   174,   174,   175,   175,   176,   176,   177,
     177,   178,   178,   179,   179,   180,   180,   181,   181,   182,
     182,   183,   183,   184,   184,   185,   185,   186,   186,   187,
     187,   188,   188,   189,   189,   190,   190,   191,   191,   192,
     192,   192,   193,   193,   194,   194,   195,   195,   196,   196,
     197,   197,   197,   198,   198,   198,   199,   199,   200,   200,
     201,   201,   201,   202,   202,   203,   203,   204,   204,   205,
     205,   206,   206,   206,   207,   207,   207,   208,   208,   208,
     209,   209,   209,   210,   210,   210,   211,   211,   211,   212,
     212,   212,   213,   213,   213,   214,   214,   214,   215,   215,
     215,   216,   216,   216,   217,   217,   217,   218,   218,   218,
     219,   219,   220,   220,   221,   221,   222,   222,   223,   223,
     224,   224,   225,   225,   225,   226,   226,   227,   227,   227,
     228,   228,   228,   229,   229,   229,   230,   230,   230,   231,
     231,   232,   232,   233,   233,   234,   234,   235,   235,   236,
     236,   237,   237,   238,   238,   239,   239,   240,   240,   240,
     241,   241,   241,   241,   241,   242,   242,   242,   242,   242,
     243,   243,   243,   243,   243,   244,   244,   244,   245,   245,
     246,   246,   247,   247,   248,   248,   249,   249,   250,   250,
     251,   251,   252,   252,   253,   253,   253,   254,   254,   254,
     255,   255,   255,   256,   256,   256,   257,   257,   258,   258,
     259,   259,   260,   260,   261,   261,   262,   262,   262,   263,
     263,   263,   264,   264,   265,   265,   266,   266,   267,   267,
     268,   268,   269,   269,   270,   270,   271,   271,   271,   272,
     272,   272,   273,   273,   274,   274
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     1,     2,     3,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     3,     4,     4,     5,     3,     4,     3,     4,     3,
       3,     2,     3,     3,     4,     1,     2,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     6,     1,     3,     3,
       3,     2,     2,     4,     5,     3,     4,     2,     3,     2,
       1,     0,     4,     2,     3,     2,     2,     2,     1,     3,
       1,     3,     2,     0,     1,     3,     1,     1,     0,     1,
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
       1,     4,     2,     4,     2,     2,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       4,     4,     2,     4,     4,     2,     4,     2,     4,     2,
       2,     1,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     4,     4,     2,     6,     6,     2,     6,     6,     2,
       6,     6,     2,     4,     4,     2,     4,     4,     2,     6,
       6,     2,     6,     6,     2,     6,     6,     2,     4,     4,
       2,     6,     6,     2,     6,     6,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     4,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     4,     4,     2,
       4,     4,     2,     4,     4,     2,     2,     2,     2,     1,
       2,     1,     2,     2,     2,     1,     2,     2,     2,     1,
       2,     6,     2,     6,     2,     3,     5,     4,     4,     2,
       4,     4,     4,     4,     2,     4,     4,     4,     4,     2,
       4,     4,     4,     4,     2,     2,     2,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     2,     2,     4,     4,     2,     4,     4,     2,
       4,     4,     2,     4,     4,     2,     4,     2,     4,     2,
       4,     2,     2,     2,     2,     2,     6,     6,     2,     4,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,     0,     4,    29,     0,     0,     0,     0,
       0,     0,     0,    45,     0,     0,     0,     0,     0,     0,
      70,     0,     0,     0,     0,     0,     0,     0,    57,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     251,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     3,
       0,     0,     0,    90,   153,    91,    92,    98,    93,    99,
      94,   100,    95,   101,    96,    97,   104,   105,   106,   107,
     108,   109,   147,   102,   103,   110,   111,   141,   142,   139,
     140,   112,   113,   114,   175,   115,   116,   117,   118,   119,
     120,   121,   122,   123,   125,   124,   126,   127,   128,   129,
     130,   131,   132,   133,   150,   134,   135,   136,   137,   138,
     149,   143,   145,   144,   146,   148,   154,   151,   152,   174,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   165,
     166,   164,   167,   168,   169,   170,   171,   172,   173,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
     187,   188,   189,   190,     9,    88,     0,     0,    88,    88,
      88,    41,    88,     0,     0,    88,    67,    69,    73,    88,
      30,    46,    14,    16,    17,    18,    19,     0,    15,     0,
       0,     0,     0,     0,     0,    75,    76,    77,    78,    61,
      80,    62,    71,     0,    82,   192,     0,   196,   195,   198,
      26,    22,    25,    24,    10,     0,    11,    12,    13,    23,
     202,     0,   206,     0,   210,     0,   214,     0,   216,     0,
     200,     0,   204,     0,   208,     0,   212,     0,   381,     0,
     233,   232,   235,   234,   237,   236,   239,   238,   330,   334,
     333,   332,   325,     0,   252,   250,   254,   253,   256,   255,
     263,     0,   266,     0,   269,     0,   272,     0,   275,     0,
     278,     0,   281,     0,   284,     0,   287,     0,   290,     0,
     293,     0,     0,   296,     0,     0,   299,   298,   297,   301,
     300,   260,   259,   303,   302,   305,   304,   307,     0,   309,
     308,   314,   312,   313,   319,     0,     0,   322,     0,     0,
     316,   315,   218,     0,   220,     0,   224,     0,   222,     0,
     228,     0,   226,     0,   247,     0,   249,     0,   242,     0,
     245,     0,   342,     0,   344,     0,   336,   231,   229,   230,
     328,   326,   327,   338,   337,   311,   310,   194,     0,   349,
       0,     0,   354,     0,     0,   359,     0,     0,   364,     0,
       0,   367,   365,   366,   379,     0,   369,     0,   371,     0,
     373,     0,   375,     0,   377,     0,   383,   382,   386,     0,
     389,     0,   392,     0,   395,     0,   397,     0,   399,     0,
     401,     0,   340,   258,   257,   403,   402,   405,   404,   408,
       0,     0,   411,     0,     0,   413,     0,   415,     0,   417,
       0,   419,     0,   421,     0,   423,     0,   425,     0,   428,
      28,    27,   427,   426,   431,   430,   429,   433,   432,   435,
     434,     5,     7,     0,     8,    89,    31,    20,    88,    21,
      35,    37,    39,    40,    42,    43,    59,    58,    60,    65,
      68,    74,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    83,    88,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,    32,    33,    36,    38,    44,    66,    47,    48,
      49,    50,    55,    52,     0,    53,    54,    51,    79,    81,
      87,    86,    72,    84,    63,   191,   197,   201,   205,   209,
     213,   215,   199,   203,   207,   211,   380,   323,   324,   261,
     262,     0,     0,     0,     0,     0,     0,   273,   274,   276,
     277,     0,     0,     0,     0,     0,     0,   288,   289,     0,
       0,     0,     0,   306,   317,   318,   320,   321,   217,   219,
     223,   221,   227,   225,   246,   248,   241,   240,   244,   243,
       0,     0,   193,   348,   347,   350,   351,   352,   353,   355,
     356,   357,   358,   360,   361,   362,   363,   378,   368,   370,
     372,   374,   376,   385,   384,   388,   387,   391,   390,   394,
     393,   396,   398,   400,     0,     0,   409,   410,   412,   414,
     416,   418,   420,   422,   424,    34,     0,     0,    64,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    56,
      85,   264,   265,   267,   268,   270,   271,   279,   280,   282,
     283,   285,   286,   292,   291,   294,   295,     0,   341,   343,
     406,   407,     0,     0,   345,     0,     0,   346
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,   139,   295,   296,   297,   268,   641,   298,   299,
     503,   140,   542,   279,   281,   141,   642,   643,   516,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
     173,   174,   175,   176,   177,   178,   179,   180,   181,   182,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   199,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   788,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   241,
     242,   243
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -245
static const yytype_int16 yypact[] =
{
    -245,   301,  -245,    25,  -245,  -245,     4,    10,    60,   126,
     139,   148,   152,  -245,   161,   162,   165,   179,   208,    -5,
    -245,   229,   253,   269,   272,  1427,  1427,   897,  -245,   145,
     200,   215,   239,   240,   245,   255,   289,    17,   133,    79,
     158,   172,   183,   195,   207,   302,   693,   781,  1093,  1104,
    1114,  1122,   222,   244,    32,   453,   106,     5,   465,   476,
    -245,  1132,  1140,  1150,  1161,  1169,  1179,  1187,  1196,  1208,
    1216,  1226,    15,    26,    28,    16,  1235,    53,    64,    93,
    1082,   111,   474,   485,   525,   619,   643,   654,   668,   703,
     717,  1243,    66,   132,   143,   167,   235,   185,   487,   489,
     257,   728,  1255,   541,   116,   545,   615,   631,  1263,   742,
     753,  1272,  1282,  1290,  1302,  1311,  1319,  1329,  1337,   932,
     943,   954,   259,  1348,  1358,  1366,   584,   642,   965,   982,
     993,  1004,  1015,  1032,  1043,    30,   505,  1054,  1065,  -245,
     294,   780,   297,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,   263,  1379,  1379,   263,   263,
     263,   266,   263,   233,   268,   263,   300,  -245,  -245,   263,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,   339,  -245,   344,
     345,    91,   348,   451,   452,  -245,  -245,  -245,  -245,   462,
    -245,   464,  -245,   466,  -245,  -245,   467,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,   468,  -245,  -245,  -245,  -245,
    -245,   471,  -245,   477,  -245,   478,  -245,   480,  -245,   482,
    -245,   538,  -245,   547,  -245,   550,  -245,   560,  -245,   561,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,   570,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,   571,  -245,   581,  -245,   585,  -245,   597,  -245,   598,
    -245,   599,  -245,   613,  -245,   614,  -245,   620,  -245,   622,
    -245,   625,   626,  -245,   640,   647,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,   648,  -245,
    -245,  -245,  -245,  -245,  -245,   651,   667,  -245,   669,   678,
    -245,  -245,  -245,   700,  -245,   704,  -245,   714,  -245,   716,
    -245,   718,  -245,   721,  -245,   722,  -245,   723,  -245,   735,
    -245,   736,  -245,   739,  -245,   748,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,   752,  -245,
     763,   773,  -245,   774,   788,  -245,   799,   806,  -245,   811,
     812,  -245,  -245,  -245,  -245,   816,  -245,   827,  -245,   929,
    -245,   930,  -245,   933,  -245,   935,  -245,  -245,  -245,   936,
    -245,   937,  -245,   938,  -245,   942,  -245,   945,  -245,   946,
    -245,   947,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
     949,   959,  -245,   960,   962,  -245,   963,  -245,   964,  -245,
     967,  -245,   968,  -245,  1010,  -245,  1013,  -245,  1014,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,   491,  -245,  -245,   490,  -245,   263,  -245,
     607,   608,  -245,  -245,  -245,   682,  -245,  -245,  -245,   908,
    -245,  -245,    11,   125,   729,   754,   666,   966,  1016,  1018,
     898,   984,   432,   263,   416,   416,   416,   416,   416,   416,
     416,   416,   416,   416,   416,   914,   171,  1374,  1374,  1374,
    1374,  1374,  1374,  1374,  1374,  1374,  1374,   416,   416,  1061,
    1026,   914,   416,   416,   416,   416,   416,   416,   416,   416,
     416,   416,  1427,  1427,   552,   578,  1427,  1427,   416,   416,
     416,   -32,    42,    43,   115,   121,   122,   914,   416,   416,
     914,   914,   914,   764,  1403,  1411,  1419,   914,   914,   914,
    1030,  1031,  1033,  1034,   416,   416,   416,   416,   416,   416,
     416,  -245,  -245,  1076,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  1113,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  1115,  -245,  1079,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  1117,  1129,  1130,  1135,  1137,  1139,  -245,  -245,  -245,
    -245,  1141,  1147,  1149,  1151,  1158,  1159,  -245,  -245,  1164,
    1167,  1168,  1176,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    1177,  1182,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  1185,  1186,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  1170,   432,  -245,  1427,
    1427,   416,   416,   416,   416,  1427,  1427,   416,   416,  1427,
    1427,   416,   416,  1103,  1156,  1157,  1157,   416,   416,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  1197,  1198,  1198,
    -245,  -245,  1188,  1174,  -245,  1205,  1207,  -245
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -245,  -245,  -245,    61,    -8,   -25,  -245,  -242,  -244,   875,
    1019,  -245,  -245,  -245,  -245,  -245,  -245,   454,  -200,  1078,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,   455,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -340
static const yytype_int16 yytable[] =
{
     267,   269,   519,   519,   518,   520,   332,   705,   245,   519,
     706,   527,   286,   288,   246,   628,   360,   369,   285,   361,
     311,   313,   315,   317,   319,   321,   323,   363,   244,   366,
     364,   499,   367,   328,   257,  -329,   339,   341,   343,   345,
     347,   349,   351,   353,   355,   357,   359,   333,   521,   522,
     523,   372,   525,   629,   373,   529,   262,   362,   370,   531,
     263,   264,   265,   266,   247,   375,   405,   406,   365,   262,
     368,   378,   380,   263,   264,   265,   266,   428,   500,   501,
     289,   707,   709,   445,   708,   710,   451,   453,   455,   457,
     459,   461,   463,   465,   377,   374,   535,   536,   474,   476,
     478,   301,   303,   305,   307,   309,   376,   331,   407,  -331,
     502,   505,   381,   325,   327,   382,   330,   432,   262,   335,
     337,   290,   263,   264,   265,   266,   291,   292,   293,   630,
     248,   294,   262,   408,   287,   290,   263,   264,   265,   266,
     291,   292,   293,   249,   410,   391,   393,   395,   397,   399,
     401,   403,   250,   383,   711,   433,   251,   712,   434,   300,
     713,   715,   426,   714,   716,   252,   253,   631,   412,   254,
     447,   449,   262,   302,   409,   657,   263,   264,   265,   266,
     467,   469,   471,   255,   304,   411,   416,   275,  -335,   486,
     488,   490,   492,   494,   496,   498,   306,   262,   508,   510,
     290,   263,   264,   265,   266,   291,   292,   293,   308,   413,
     294,   262,   256,   658,   290,   263,   264,   265,   266,   291,
     292,   293,   262,   324,   294,   290,   263,   264,   265,   266,
     291,   292,   293,   258,   262,   294,   414,   290,   263,   264,
     265,   266,   291,   292,   293,   326,   262,   294,   276,   290,
     263,   264,   265,   266,   291,   292,   293,   259,   423,   294,
     472,   262,  -339,   277,   290,   263,   264,   265,   266,   291,
     292,   293,   526,   260,   294,   290,   261,   415,   278,   280,
     291,   292,   293,   262,   282,   517,   290,   263,   264,   265,
     266,   291,   292,   293,   283,   284,   294,   511,   519,   424,
     514,     2,     3,   310,     4,   515,   524,   528,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,   623,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
     530,   262,    36,   644,   532,   263,   264,   265,   266,   533,
     534,    37,    38,   537,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   329,   262,   538,   539,   290,   263,
     264,   265,   266,   291,   292,   293,   334,   540,   294,   541,
     543,   640,   544,   545,   290,   384,   546,   336,   385,   291,
     292,   293,   547,   548,   517,   549,   387,   550,   417,   388,
     420,   418,   262,   421,   621,   290,   263,   264,   265,   266,
     291,   292,   293,   519,   262,   294,   504,   290,   263,   264,
     265,   266,   291,   292,   293,   262,   386,   294,   290,   263,
     264,   265,   266,   291,   292,   293,   390,   389,   294,   419,
     622,   422,   659,   661,   663,   665,   667,   669,   671,   673,
     675,   677,   429,   551,   262,   430,   435,   656,   263,   264,
     265,   266,   552,   500,   501,   553,   696,   694,   695,   697,
     699,   700,   701,   683,   262,   554,   555,   290,   263,   264,
     265,   266,   291,   292,   293,   556,   557,   294,   724,   726,
     728,   730,   698,   431,   436,   479,   558,   437,   480,   717,
     559,   262,   720,   721,   722,   263,   264,   265,   266,   731,
     732,   733,   560,   561,   562,   645,   646,   647,   648,   649,
     650,   651,   652,   653,   654,   655,   438,   262,   563,   564,
     392,   263,   264,   265,   266,   565,   481,   566,   679,   680,
     567,   568,   441,   684,   685,   686,   687,   688,   689,   690,
     691,   692,   693,   482,   394,   569,   483,   624,   625,   702,
     703,   704,   570,   571,   439,   396,   572,   440,   262,   718,
     719,   290,   263,   264,   265,   266,   291,   292,   293,   398,
     442,   294,   573,   443,   574,   738,   739,   740,   741,   742,
     743,   744,   262,   575,   484,   290,   263,   264,   265,   266,
     291,   292,   293,   262,   312,   294,   290,   263,   264,   265,
     266,   291,   292,   293,   400,   576,   294,   262,   634,   577,
     290,   263,   264,   265,   266,   291,   292,   293,   402,   578,
     294,   579,   626,   580,   771,   772,   581,   582,   583,   425,
     777,   778,   262,   632,   781,   782,   263,   264,   265,   266,
     584,   585,   262,   446,   586,   290,   263,   264,   265,   266,
     291,   292,   293,   587,   448,   294,   262,   588,   633,   290,
     263,   264,   265,   266,   291,   292,   293,   262,   589,   294,
     290,   263,   264,   265,   266,   291,   292,   293,   590,   591,
     294,   262,   314,   512,   290,   263,   264,   265,   266,   291,
     292,   293,   262,   592,   294,   290,   263,   264,   265,   266,
     291,   292,   293,   262,   593,   294,   723,   263,   264,   265,
     266,   594,   773,   774,   775,   776,   595,   596,   779,   780,
     262,   597,   783,   784,   263,   264,   265,   266,   790,   791,
      37,    38,   598,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   466,   599,   600,   270,   638,   601,   271,
     602,   603,   604,   605,   468,   272,   273,   606,   627,   274,
     607,   608,   609,   262,   610,   470,   290,   263,   264,   265,
     266,   291,   292,   293,   611,   612,   485,   613,   614,   615,
     635,   262,   616,   617,   290,   263,   264,   265,   266,   291,
     292,   293,   262,   487,   294,   290,   263,   264,   265,   266,
     291,   292,   293,   262,   489,   294,   290,   263,   264,   265,
     266,   291,   292,   293,   262,   491,   294,   290,   263,   264,
     265,   266,   291,   292,   293,   618,   493,   294,   619,   620,
     636,   262,   637,   639,   290,   263,   264,   265,   266,   291,
     292,   293,   262,   495,   294,   290,   263,   264,   265,   266,
     291,   292,   293,   262,   497,   294,   290,   263,   264,   265,
     266,   291,   292,   293,   262,   507,   294,   290,   263,   264,
     265,   266,   291,   292,   293,   681,   509,   294,   682,   734,
     735,   262,   736,   737,   290,   263,   264,   265,   266,   291,
     292,   293,   262,   379,   294,   290,   263,   264,   265,   266,
     291,   292,   293,   262,   316,   294,   290,   263,   264,   265,
     266,   291,   292,   293,   262,   318,   294,   290,   263,   264,
     265,   266,   291,   292,   293,   320,   745,   294,   746,   748,
     747,   262,   749,   322,   290,   263,   264,   265,   266,   291,
     292,   293,   262,   338,   750,   751,   263,   264,   265,   266,
     752,   340,   753,   262,   754,   785,   755,   263,   264,   265,
     266,   342,   756,   262,   757,   506,   758,   263,   264,   265,
     266,   262,   344,   759,   760,   263,   264,   265,   266,   761,
     346,   262,   762,   763,   769,   263,   264,   265,   266,   262,
     348,   764,   765,   263,   264,   265,   266,   766,   350,   262,
     767,   768,   794,   263,   264,   265,   266,   352,   786,   787,
     262,   770,   792,   793,   263,   264,   265,   266,   262,   354,
     796,   797,   263,   264,   265,   266,   795,   356,   262,   513,
       0,   789,   263,   264,   265,   266,   262,   358,     0,     0,
     263,   264,   265,   266,     0,   262,   371,     0,     0,   263,
     264,   265,   266,     0,   404,     0,     0,   262,     0,     0,
       0,   263,   264,   265,   266,   262,   427,     0,     0,   263,
     264,   265,   266,     0,   444,   262,     0,     0,     0,   263,
     264,   265,   266,   450,   262,     0,     0,     0,   263,   264,
     265,   266,   262,   452,     0,     0,   263,   264,   265,   266,
       0,   454,     0,     0,   262,     0,     0,     0,   263,   264,
     265,   266,   262,   456,     0,     0,   263,   264,   265,   266,
       0,   262,   458,     0,     0,   263,   264,   265,   266,     0,
     460,   262,     0,     0,     0,   263,   264,   265,   266,   262,
     462,     0,     0,   263,   264,   265,   266,     0,   464,     0,
       0,   262,     0,     0,     0,   263,   264,   265,   266,   473,
     262,     0,     0,     0,   263,   264,   265,   266,   262,   475,
       0,     0,   263,   264,   265,   266,     0,   477,   262,     0,
       0,     0,   263,   264,   265,   266,   262,     0,     0,     0,
     263,   264,   265,   266,     0,     0,     0,   262,     0,     0,
       0,   263,   264,   265,   266,     0,     0,   262,     0,     0,
       0,   263,   264,   265,   266,   262,     0,     0,     0,   263,
     264,   265,   266,   262,     0,     0,   290,   263,   264,   265,
     266,   290,   292,   293,     0,     0,   291,   292,   293,     0,
       0,   517,   660,   662,   664,   666,   668,   670,   672,   674,
     676,   678,   262,     0,     0,   725,   263,   264,   265,   266,
     262,     0,     0,   727,   263,   264,   265,   266,   262,     0,
       0,   729,   263,   264,   265,   266,   262,     0,     0,     0,
     263,   264,   265,   266
};

static const yytype_int16 yycheck[] =
{
      25,    26,   246,   247,   246,   247,     1,    39,     4,   253,
      42,   253,    37,    38,     4,     4,     1,     1,     1,     4,
      45,    46,    47,    48,    49,    50,    51,     1,     3,     1,
       4,     1,     4,     1,    39,     3,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    42,   248,   249,
     250,    76,   252,    42,     1,   255,    39,    42,    42,   259,
      43,    44,    45,    46,     4,     1,    91,     1,    42,    39,
      42,    79,    80,    43,    44,    45,    46,   102,    48,    49,
       1,    39,    39,   108,    42,    42,   111,   112,   113,   114,
     115,   116,   117,   118,     1,    42,     5,     6,   123,   124,
     125,    40,    41,    42,    43,    44,    42,     1,    42,     3,
     135,   136,     1,    52,    53,     4,    55,     1,    39,    58,
      59,    42,    43,    44,    45,    46,    47,    48,    49,     4,
       4,    52,    39,     1,     1,    42,    43,    44,    45,    46,
      47,    48,    49,     4,     1,    84,    85,    86,    87,    88,
      89,    90,     4,    42,    39,    39,     4,    42,    42,     1,
      39,    39,   101,    42,    42,     4,     4,    42,     1,     4,
     109,   110,    39,     1,    42,     4,    43,    44,    45,    46,
     119,   120,   121,     4,     1,    42,     1,    42,     3,   128,
     129,   130,   131,   132,   133,   134,     1,    39,   137,   138,
      42,    43,    44,    45,    46,    47,    48,    49,     1,    42,
      52,    39,     4,    42,    42,    43,    44,    45,    46,    47,
      48,    49,    39,     1,    52,    42,    43,    44,    45,    46,
      47,    48,    49,     4,    39,    52,     1,    42,    43,    44,
      45,    46,    47,    48,    49,     1,    39,    52,    48,    42,
      43,    44,    45,    46,    47,    48,    49,     4,     1,    52,
       1,    39,     3,    48,    42,    43,    44,    45,    46,    47,
      48,    49,    39,     4,    52,    42,     4,    42,    39,    39,
      47,    48,    49,    39,    39,    52,    42,    43,    44,    45,
      46,    47,    48,    49,    39,     6,    52,     3,   542,    42,
       3,     0,     1,     1,     3,    42,    40,    39,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,   518,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      40,    39,    41,   543,     5,    43,    44,    45,    46,     5,
       5,    50,    51,     5,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,     1,    39,     5,     5,    42,    43,
      44,    45,    46,    47,    48,    49,     1,     5,    52,     5,
       4,    39,     5,     5,    42,     1,     5,     1,     4,    47,
      48,    49,     5,     5,    52,     5,     1,     5,     1,     4,
       1,     4,    39,     4,     3,    42,    43,    44,    45,    46,
      47,    48,    49,   747,    39,    52,     1,    42,    43,    44,
      45,    46,    47,    48,    49,    39,    42,    52,    42,    43,
      44,    45,    46,    47,    48,    49,     1,    42,    52,    42,
      40,    42,   557,   558,   559,   560,   561,   562,   563,   564,
     565,   566,     1,     5,    39,     4,     1,   555,    43,    44,
      45,    46,     5,    48,    49,     5,     4,   582,   583,   584,
     585,   586,   587,   571,    39,     5,     5,    42,    43,    44,
      45,    46,    47,    48,    49,     5,     5,    52,   603,   604,
     605,   606,     4,    42,    39,     1,     5,    42,     4,   597,
       5,    39,   600,   601,   602,    43,    44,    45,    46,   607,
     608,   609,     5,     5,     5,   544,   545,   546,   547,   548,
     549,   550,   551,   552,   553,   554,     1,    39,     5,     5,
       1,    43,    44,    45,    46,     5,    42,     5,   567,   568,
       5,     5,     1,   572,   573,   574,   575,   576,   577,   578,
     579,   580,   581,     1,     1,     5,     4,    40,    40,   588,
     589,   590,     5,     5,    39,     1,     5,    42,    39,   598,
     599,    42,    43,    44,    45,    46,    47,    48,    49,     1,
      39,    52,     5,    42,     5,   614,   615,   616,   617,   618,
     619,   620,    39,     5,    42,    42,    43,    44,    45,    46,
      47,    48,    49,    39,     1,    52,    42,    43,    44,    45,
      46,    47,    48,    49,     1,     5,    52,    39,    42,     5,
      42,    43,    44,    45,    46,    47,    48,    49,     1,     5,
      52,     5,    40,     5,   749,   750,     5,     5,     5,     1,
     755,   756,    39,     4,   759,   760,    43,    44,    45,    46,
       5,     5,    39,     1,     5,    42,    43,    44,    45,    46,
      47,    48,    49,     5,     1,    52,    39,     5,     4,    42,
      43,    44,    45,    46,    47,    48,    49,    39,     5,    52,
      42,    43,    44,    45,    46,    47,    48,    49,     5,     5,
      52,    39,     1,     3,    42,    43,    44,    45,    46,    47,
      48,    49,    39,     5,    52,    42,    43,    44,    45,    46,
      47,    48,    49,    39,     5,    52,    42,    43,    44,    45,
      46,     5,   751,   752,   753,   754,     5,     5,   757,   758,
      39,     5,   761,   762,    43,    44,    45,    46,   767,   768,
      50,    51,     5,    53,    54,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,   107,   108,   109,
     110,   111,   112,   113,   114,   115,   116,   117,   118,   119,
     120,   121,   122,   123,   124,   125,   126,   127,   128,   129,
     130,   131,   132,   133,   134,   135,   136,   137,   138,   139,
     140,   141,   142,   143,   144,   145,   146,   147,   148,   149,
     150,   151,   152,     1,     5,     5,    39,    39,     5,    42,
       5,     5,     5,     5,     1,    48,    49,     5,    40,    52,
       5,     5,     5,    39,     5,     1,    42,    43,    44,    45,
      46,    47,    48,    49,     5,     5,     1,     5,     5,     5,
       4,    39,     5,     5,    42,    43,    44,    45,    46,    47,
      48,    49,    39,     1,    52,    42,    43,    44,    45,    46,
      47,    48,    49,    39,     1,    52,    42,    43,    44,    45,
      46,    47,    48,    49,    39,     1,    52,    42,    43,    44,
      45,    46,    47,    48,    49,     5,     1,    52,     5,     5,
       4,    39,     4,    39,    42,    43,    44,    45,    46,    47,
      48,    49,    39,     1,    52,    42,    43,    44,    45,    46,
      47,    48,    49,    39,     1,    52,    42,    43,    44,    45,
      46,    47,    48,    49,    39,     1,    52,    42,    43,    44,
      45,    46,    47,    48,    49,     4,     1,    52,    42,    39,
      39,    39,    39,    39,    42,    43,    44,    45,    46,    47,
      48,    49,    39,     1,    52,    42,    43,    44,    45,    46,
      47,    48,    49,    39,     1,    52,    42,    43,    44,    45,
      46,    47,    48,    49,    39,     1,    52,    42,    43,    44,
      45,    46,    47,    48,    49,     1,    40,    52,     5,    40,
       5,    39,     5,     1,    42,    43,    44,    45,    46,    47,
      48,    49,    39,     1,     5,     5,    43,    44,    45,    46,
       5,     1,     5,    39,     5,    42,     5,    43,    44,    45,
      46,     1,     5,    39,     5,   136,     5,    43,    44,    45,
      46,    39,     1,     5,     5,    43,    44,    45,    46,     5,
       1,    39,     5,     5,     4,    43,    44,    45,    46,    39,
       1,     5,     5,    43,    44,    45,    46,     5,     1,    39,
       5,     5,     4,    43,    44,    45,    46,     1,    42,    42,
      39,   747,     5,     5,    43,    44,    45,    46,    39,     1,
       5,     4,    43,    44,    45,    46,    42,     1,    39,   141,
      -1,   766,    43,    44,    45,    46,    39,     1,    -1,    -1,
      43,    44,    45,    46,    -1,    39,     1,    -1,    -1,    43,
      44,    45,    46,    -1,     1,    -1,    -1,    39,    -1,    -1,
      -1,    43,    44,    45,    46,    39,     1,    -1,    -1,    43,
      44,    45,    46,    -1,     1,    39,    -1,    -1,    -1,    43,
      44,    45,    46,     1,    39,    -1,    -1,    -1,    43,    44,
      45,    46,    39,     1,    -1,    -1,    43,    44,    45,    46,
      -1,     1,    -1,    -1,    39,    -1,    -1,    -1,    43,    44,
      45,    46,    39,     1,    -1,    -1,    43,    44,    45,    46,
      -1,    39,     1,    -1,    -1,    43,    44,    45,    46,    -1,
       1,    39,    -1,    -1,    -1,    43,    44,    45,    46,    39,
       1,    -1,    -1,    43,    44,    45,    46,    -1,     1,    -1,
      -1,    39,    -1,    -1,    -1,    43,    44,    45,    46,     1,
      39,    -1,    -1,    -1,    43,    44,    45,    46,    39,     1,
      -1,    -1,    43,    44,    45,    46,    -1,     1,    39,    -1,
      -1,    -1,    43,    44,    45,    46,    39,    -1,    -1,    -1,
      43,    44,    45,    46,    -1,    -1,    -1,    39,    -1,    -1,
      -1,    43,    44,    45,    46,    -1,    -1,    39,    -1,    -1,
      -1,    43,    44,    45,    46,    39,    -1,    -1,    -1,    43,
      44,    45,    46,    39,    -1,    -1,    42,    43,    44,    45,
      46,    42,    48,    49,    -1,    -1,    47,    48,    49,    -1,
      -1,    52,   557,   558,   559,   560,   561,   562,   563,   564,
     565,   566,    39,    -1,    -1,    42,    43,    44,    45,    46,
      39,    -1,    -1,    42,    43,    44,    45,    46,    39,    -1,
      -1,    42,    43,    44,    45,    46,    39,    -1,    -1,    -1,
      43,    44,    45,    46
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   154,     0,     1,     3,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    41,    50,    51,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   147,   148,   149,   150,   151,   152,   155,
     164,   168,   172,   173,   174,   175,   176,   177,   178,   179,
     180,   181,   182,   183,   184,   185,   186,   187,   188,   189,
     190,   191,   192,   193,   194,   195,   196,   197,   198,   199,
     200,   201,   202,   203,   204,   205,   206,   207,   208,   209,
     210,   211,   212,   213,   214,   215,   216,   217,   218,   219,
     220,   221,   222,   223,   224,   225,   226,   227,   228,   229,
     230,   231,   232,   233,   234,   235,   236,   237,   238,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,     3,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,    39,     4,     4,
       4,     4,    39,    43,    44,    45,    46,   158,   159,   158,
      39,    42,    48,    49,    52,    42,    48,    48,    39,   166,
      39,   167,    39,    39,     6,     1,   158,     1,   158,     1,
      42,    47,    48,    49,    52,   156,   157,   158,   161,   162,
       1,   156,     1,   156,     1,   156,     1,   156,     1,   156,
       1,   158,     1,   158,     1,   158,     1,   158,     1,   158,
       1,   158,     1,   158,     1,   156,     1,   156,     1,     1,
     156,     1,     1,    42,     1,   156,     1,   156,     1,   158,
       1,   158,     1,   158,     1,   158,     1,   158,     1,   158,
       1,   158,     1,   158,     1,   158,     1,   158,     1,   158,
       1,     4,    42,     1,     4,    42,     1,     4,    42,     1,
      42,     1,   158,     1,    42,     1,    42,     1,   157,     1,
     157,     1,     4,    42,     1,     4,    42,     1,     4,    42,
       1,   156,     1,   156,     1,   156,     1,   156,     1,   156,
       1,   156,     1,   156,     1,   158,     1,    42,     1,    42,
       1,    42,     1,    42,     1,    42,     1,     1,     4,    42,
       1,     4,    42,     1,    42,     1,   156,     1,   158,     1,
       4,    42,     1,    39,    42,     1,    39,    42,     1,    39,
      42,     1,    39,    42,     1,   158,     1,   156,     1,   156,
       1,   158,     1,   158,     1,   158,     1,   158,     1,   158,
       1,   158,     1,   158,     1,   158,     1,   156,     1,   156,
       1,   156,     1,     1,   158,     1,   158,     1,   158,     1,
       4,    42,     1,     4,    42,     1,   156,     1,   156,     1,
     156,     1,   156,     1,   156,     1,   156,     1,   156,     1,
      48,    49,   158,   163,     1,   158,   163,     1,   156,     1,
     156,     3,     3,   172,     3,    42,   171,    52,   160,   161,
     160,   171,   171,   171,    40,   171,    39,   160,    39,   171,
      40,   171,     5,     5,     5,     5,     6,     5,     5,     5,
       5,     5,   165,     4,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     3,    40,   171,    40,    40,    40,    40,     4,    42,
       4,    42,     4,     4,    42,     4,     4,     4,    39,    39,
      39,   160,   169,   170,   171,   156,   156,   156,   156,   156,
     156,   156,   156,   156,   156,   156,   157,     4,    42,   158,
     162,   158,   162,   158,   162,   158,   162,   158,   162,   158,
     162,   158,   162,   158,   162,   158,   162,   158,   162,   156,
     156,     4,    42,   157,   156,   156,   156,   156,   156,   156,
     156,   156,   156,   156,   158,   158,     4,   158,     4,   158,
     158,   158,   156,   156,   156,    39,    42,    39,    42,    39,
      42,    39,    42,    39,    42,    39,    42,   157,   156,   156,
     157,   157,   157,    42,   158,    42,   158,    42,   158,    42,
     158,   157,   157,   157,    39,    39,    39,    39,   156,   156,
     156,   156,   156,   156,   156,    40,     5,     5,    40,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     4,
     170,   158,   158,   156,   156,   156,   156,   158,   158,   156,
     156,   158,   158,   156,   156,    42,    42,    42,   239,   239,
     156,   156,     5,     5,     4,    42,     5,     4
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
#line 241 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax, LINE - 1 ); ;}
    break;

  case 29:
#line 255 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addEntry(); ;}
    break;

  case 30:
#line 256 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->setModuleName( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 31:
#line 257 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 32:
#line 258 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 33:
#line 259 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 34:
#line 260 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); ;}
    break;

  case 35:
#line 261 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 36:
#line 262 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 37:
#line 263 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 38:
#line 264 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 39:
#line 265 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLocal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 40:
#line 266 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addParam( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 41:
#line 267 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 42:
#line 268 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (3)]), true ); ;}
    break;

  case 43:
#line 269 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 44:
#line 270 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 45:
#line 271 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); ;}
    break;

  case 46:
#line 272 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 47:
#line 273 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 48:
#line 274 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 49:
#line 275 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); ;}
    break;

  case 50:
#line 276 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); ;}
    break;

  case 51:
#line 277 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 52:
#line 278 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 53:
#line 279 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 54:
#line 280 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 55:
#line 281 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 56:
#line 282 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (6)]), (yyvsp[(6) - (6)]), (yyvsp[(4) - (6)]) ); ;}
    break;

  case 57:
#line 283 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDEndSwitch(); ;}
    break;

  case 58:
#line 284 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 59:
#line 285 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 60:
#line 286 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addPropRef( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 63:
#line 289 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 64:
#line 290 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); ;}
    break;

  case 65:
#line 291 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 66:
#line 292 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 67:
#line 293 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 68:
#line 294 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (3)]), true ); ;}
    break;

  case 69:
#line 295 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassCtor( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 70:
#line 296 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); /* Currently the same as .endfunc */ ;}
    break;

  case 71:
#line 297 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInherit((yyvsp[(2) - (2)])); ;}
    break;

  case 73:
#line 298 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFrom( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 74:
#line 299 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addExtern( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 75:
#line 300 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addDLine( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 76:
#line 302 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      ;}
    break;

  case 77:
#line 307 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      ;}
    break;

  case 78:
#line 315 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(1) - (1)]) ); ;}
    break;

  case 79:
#line 316 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(3) - (3)]) ); ;}
    break;

  case 80:
#line 320 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(1) - (1)]) ); ;}
    break;

  case 81:
#line 321 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(3) - (3)]) ); ;}
    break;

  case 82:
#line 324 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->defineLabel( (yyvsp[(1) - (2)])->asLabel() ); ;}
    break;

  case 84:
#line 328 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(1) - (1)]) ); ;}
    break;

  case 85:
#line 329 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(3) - (3)]) ); ;}
    break;

  case 88:
#line 338 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {(yyval) = new Falcon::Pseudo( LINE, (Falcon::int64) 0 ); ;}
    break;

  case 191:
#line 447 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 192:
#line 448 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LD" ); ;}
    break;

  case 193:
#line 452 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDRF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 194:
#line 453 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDRF" ); ;}
    break;

  case 195:
#line 457 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LNIL, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 196:
#line 458 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LNIL" ); ;}
    break;

  case 197:
#line 462 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 198:
#line 463 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADD" ); ;}
    break;

  case 199:
#line 467 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 200:
#line 468 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADDS" ); ;}
    break;

  case 201:
#line 473 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 202:
#line 474 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUB" ); ;}
    break;

  case 203:
#line 478 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUBS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 204:
#line 479 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUBS" ); ;}
    break;

  case 205:
#line 483 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MUL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 206:
#line 484 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MUL" ); ;}
    break;

  case 207:
#line 488 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MULS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 208:
#line 489 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MULS" ); ;}
    break;

  case 209:
#line 494 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 210:
#line 495 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIV" ); ;}
    break;

  case 211:
#line 499 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 212:
#line 500 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIVS" ); ;}
    break;

  case 213:
#line 504 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MOD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 214:
#line 505 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MOD" ); ;}
    break;

  case 215:
#line 509 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POW, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 216:
#line 510 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POW" ); ;}
    break;

  case 217:
#line 515 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 218:
#line 516 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EQ" ); ;}
    break;

  case 219:
#line 520 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 220:
#line 521 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEQ" ); ;}
    break;

  case 221:
#line 525 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 222:
#line 526 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GE" ); ;}
    break;

  case 223:
#line 530 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 224:
#line 531 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GT" ); ;}
    break;

  case 225:
#line 535 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 226:
#line 536 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LE" ); ;}
    break;

  case 227:
#line 540 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 228:
#line 541 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LT" ); ;}
    break;

  case 229:
#line 545 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); ;}
    break;

  case 230:
#line 546 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); ;}
    break;

  case 231:
#line 547 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRY" ); ;}
    break;

  case 232:
#line 551 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INC, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 233:
#line 552 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INC" ); ;}
    break;

  case 234:
#line 556 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DEC, (yyvsp[(2) - (2)])  ); ;}
    break;

  case 235:
#line 557 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DEC" ); ;}
    break;

  case 236:
#line 561 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEG, (yyvsp[(2) - (2)])  ); ;}
    break;

  case 237:
#line 562 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEG" ); ;}
    break;

  case 238:
#line 566 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOT, (yyvsp[(2) - (2)])  ); ;}
    break;

  case 239:
#line 567 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOT" ); ;}
    break;

  case 240:
#line 571 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 241:
#line 572 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 242:
#line 573 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "CALL" ); ;}
    break;

  case 243:
#line 577 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 244:
#line 578 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 245:
#line 579 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INST" ); ;}
    break;

  case 246:
#line 583 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_UNPK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 247:
#line 584 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPK" ); ;}
    break;

  case 248:
#line 588 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_UNPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 249:
#line 589 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPS" ); ;}
    break;

  case 250:
#line 594 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PUSH, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 251:
#line 595 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHN ); ;}
    break;

  case 252:
#line 596 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PUSH" ); ;}
    break;

  case 253:
#line 600 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHR, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 254:
#line 601 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSHR" ); ;}
    break;

  case 255:
#line 606 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_POP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 256:
#line 607 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POP" ); ;}
    break;

  case 257:
#line 611 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PEEK, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 258:
#line 612 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PEEK" ); ;}
    break;

  case 259:
#line 616 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XPOP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 260:
#line 617 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XPOP" ); ;}
    break;

  case 261:
#line 622 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 262:
#line 623 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 263:
#line 624 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDV" ); ;}
    break;

  case 264:
#line 628 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 265:
#line 629 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 266:
#line 630 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVT" ); ;}
    break;

  case 267:
#line 634 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 268:
#line 635 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 269:
#line 636 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STV" ); ;}
    break;

  case 270:
#line 640 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 271:
#line 641 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 272:
#line 642 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVR" ); ;}
    break;

  case 273:
#line 646 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 274:
#line 647 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 275:
#line 648 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVS" ); ;}
    break;

  case 276:
#line 652 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 277:
#line 653 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 278:
#line 654 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDP" ); yyerrok; ;}
    break;

  case 279:
#line 658 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 280:
#line 659 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 281:
#line 660 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPT" ); yyerrok; ;}
    break;

  case 282:
#line 664 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 283:
#line 665 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 284:
#line 666 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STP" ); ;}
    break;

  case 285:
#line 670 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 286:
#line 671 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 287:
#line 672 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPR" ); ;}
    break;

  case 288:
#line 676 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 289:
#line 677 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 290:
#line 678 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPS" ); ;}
    break;

  case 291:
#line 682 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 292:
#line 683 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 293:
#line 684 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAV" ); ;}
    break;

  case 294:
#line 688 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 295:
#line 689 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 296:
#line 690 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAN" ); ;}
    break;

  case 297:
#line 694 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 298:
#line 695 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 299:
#line 696 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAL" ); ;}
    break;

  case 300:
#line 700 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_IPOP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 301:
#line 701 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IPOP" ); ;}
    break;

  case 302:
#line 705 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GENA, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 303:
#line 706 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENA" ); ;}
    break;

  case 304:
#line 710 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GEND, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 305:
#line 711 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEND" ); ;}
    break;

  case 306:
#line 715 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GENR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 307:
#line 716 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENR" ); ;}
    break;

  case 308:
#line 720 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GEOR, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 309:
#line 721 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEOR" ); ;}
    break;

  case 310:
#line 725 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RIS, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 311:
#line 726 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RIS" ); ;}
    break;

  case 312:
#line 730 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 313:
#line 731 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 314:
#line 732 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JMP" ); ;}
    break;

  case 315:
#line 736 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOOL, (yyvsp[(1) - (2)]) ); ;}
    break;

  case 316:
#line 737 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOOL" ); ;}
    break;

  case 317:
#line 741 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 318:
#line 742 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 319:
#line 743 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFT" ); ;}
    break;

  case 320:
#line 747 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 321:
#line 748 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 322:
#line 749 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFF" ); ;}
    break;

  case 323:
#line 754 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 324:
#line 755 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 325:
#line 756 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORK" ); ;}
    break;

  case 326:
#line 760 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 327:
#line 761 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 328:
#line 762 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JTRY" ); ;}
    break;

  case 329:
#line 766 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RET ); ;}
    break;

  case 330:
#line 767 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RET" ); ;}
    break;

  case 331:
#line 771 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETA ); ;}
    break;

  case 332:
#line 772 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETA" ); ;}
    break;

  case 333:
#line 776 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETV, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 334:
#line 777 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETV" ); ;}
    break;

  case 335:
#line 781 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOP ); ;}
    break;

  case 336:
#line 782 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOP" ); ;}
    break;

  case 337:
#line 786 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_PTRY, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 338:
#line 787 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PTRY" ); ;}
    break;

  case 339:
#line 791 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_END ); ;}
    break;

  case 340:
#line 792 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "END" ); ;}
    break;

  case 341:
#line 796 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 342:
#line 797 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SWCH" ); ;}
    break;

  case 343:
#line 801 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 344:
#line 802 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SELE" ); ;}
    break;

  case 345:
#line 807 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         Falcon::Pseudo *psd = new Falcon::Pseudo( Falcon::Pseudo::tswitch_list );
         psd->line( LINE );
         psd->asList()->pushBack( (yyvsp[(1) - (3)]) );
         psd->asList()->pushBack( (yyvsp[(3) - (3)]) );
         (yyval) = psd;
      ;}
    break;

  case 346:
#line 816 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(3) - (5)]) );
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(5) - (5)]) );
         (yyval) = (yyvsp[(1) - (5)]);
      ;}
    break;

  case 347:
#line 824 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); ;}
    break;

  case 348:
#line 825 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); ;}
    break;

  case 349:
#line 826 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ONCE" ); ;}
    break;

  case 350:
#line 830 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 351:
#line 831 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 352:
#line 832 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 353:
#line 833 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 354:
#line 834 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BAND" ); ;}
    break;

  case 355:
#line 838 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 356:
#line 839 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 357:
#line 840 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 358:
#line 841 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 359:
#line 842 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOR" ); ;}
    break;

  case 360:
#line 846 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 361:
#line 847 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 362:
#line 848 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 363:
#line 849 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 364:
#line 850 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); ;}
    break;

  case 365:
#line 854 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 366:
#line 855 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 367:
#line 856 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); ;}
    break;

  case 368:
#line 860 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_AND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 369:
#line 861 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "AND" ); ;}
    break;

  case 370:
#line 865 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_OR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 371:
#line 866 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "OR" ); ;}
    break;

  case 372:
#line 870 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ANDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 373:
#line 871 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ANDS" ); ;}
    break;

  case 374:
#line 875 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 375:
#line 876 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ORS" ); ;}
    break;

  case 376:
#line 880 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 377:
#line 881 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XORS" ); ;}
    break;

  case 378:
#line 885 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MODS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 379:
#line 886 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MODS" ); ;}
    break;

  case 380:
#line 890 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POWS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 381:
#line 891 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POWS" ); ;}
    break;

  case 382:
#line 895 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOTS, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 383:
#line 896 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOTS" ); ;}
    break;

  case 384:
#line 900 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 385:
#line 901 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 386:
#line 902 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HAS" ); ;}
    break;

  case 387:
#line 906 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 388:
#line 907 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 389:
#line 908 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HASN" ); ;}
    break;

  case 390:
#line 912 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 391:
#line 913 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 392:
#line 914 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVE" ); ;}
    break;

  case 393:
#line 918 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 394:
#line 919 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 395:
#line 920 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVN" ); ;}
    break;

  case 396:
#line 925 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_IN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 397:
#line 926 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IN" ); ;}
    break;

  case 398:
#line 930 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOIN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 399:
#line 931 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOIN" ); ;}
    break;

  case 400:
#line 935 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PROV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 401:
#line 936 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PROV" ); ;}
    break;

  case 402:
#line 940 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSIN, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 403:
#line 941 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSIN" ); ;}
    break;

  case 404:
#line 945 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PASS, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 405:
#line 946 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PASS" ); ;}
    break;

  case 406:
#line 950 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_FORI, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 407:
#line 951 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_FORI, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 408:
#line 952 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORI" ); ;}
    break;

  case 409:
#line 956 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_FORN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 410:
#line 957 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_FORN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 411:
#line 958 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORN" ); ;}
    break;

  case 412:
#line 962 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 413:
#line 963 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHR" ); ;}
    break;

  case 414:
#line 967 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 415:
#line 968 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHL" ); ;}
    break;

  case 416:
#line 972 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHRS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 417:
#line 973 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHRS" ); ;}
    break;

  case 418:
#line 977 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHLS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 419:
#line 978 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHLS" ); ;}
    break;

  case 420:
#line 982 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 421:
#line 983 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVR" ); ;}
    break;

  case 422:
#line 987 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 423:
#line 988 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPR" ); ;}
    break;

  case 424:
#line 992 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LSB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 425:
#line 993 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LSB" ); ;}
    break;

  case 426:
#line 997 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 427:
#line 998 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 428:
#line 999 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INDI" ); ;}
    break;

  case 429:
#line 1003 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 430:
#line 1004 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 431:
#line 1005 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "STEX" ); ;}
    break;

  case 432:
#line 1009 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_TRAC, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 433:
#line 1010 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "TRAC" ); ;}
    break;

  case 434:
#line 1014 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_WRT, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 435:
#line 1015 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "WRT" ); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 4098 "/home/gian/Progetti/falcon/core/engine/fasm_parser.cpp"
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


#line 1018 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
 /* c code */


/****************************************************
* C Code for falcon HSM compiler
*****************************************************/


void fasm_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of falcon_parser.yxx */


