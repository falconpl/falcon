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
#define yyparse flc_src_parse
#define yylex   flc_src_lex
#define yyerror flc_src_error
#define yylval  flc_src_lval
#define yychar  flc_src_char
#define yydebug flc_src_debug
#define yynerrs flc_src_nerrs


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     INTNUM = 259,
     DBLNUM = 260,
     SYMBOL = 261,
     STRING = 262,
     NIL = 263,
     END = 264,
     DEF = 265,
     WHILE = 266,
     BREAK = 267,
     CONTINUE = 268,
     DROPPING = 269,
     IF = 270,
     ELSE = 271,
     ELIF = 272,
     FOR = 273,
     FORFIRST = 274,
     FORLAST = 275,
     FORMIDDLE = 276,
     SWITCH = 277,
     CASE = 278,
     DEFAULT = 279,
     SELECT = 280,
     SENDER = 281,
     SELF = 282,
     GIVE = 283,
     TRY = 284,
     CATCH = 285,
     RAISE = 286,
     CLASS = 287,
     FROM = 288,
     OBJECT = 289,
     RETURN = 290,
     GLOBAL = 291,
     LAMBDA = 292,
     INIT = 293,
     LOAD = 294,
     LAUNCH = 295,
     CONST_KW = 296,
     ATTRIBUTES = 297,
     PASS = 298,
     EXPORT = 299,
     DIRECTIVE = 300,
     COLON = 301,
     FUNCDECL = 302,
     STATIC = 303,
     FORDOT = 304,
     LISTPAR = 305,
     LOOP = 306,
     TRUE_TOKEN = 307,
     FALSE_TOKEN = 308,
     OUTER_STRING = 309,
     CLOSEPAR = 310,
     OPENPAR = 311,
     CLOSESQUARE = 312,
     OPENSQUARE = 313,
     DOT = 314,
     ASSIGN_POW = 315,
     ASSIGN_SHL = 316,
     ASSIGN_SHR = 317,
     ASSIGN_BXOR = 318,
     ASSIGN_BOR = 319,
     ASSIGN_BAND = 320,
     ASSIGN_MOD = 321,
     ASSIGN_DIV = 322,
     ASSIGN_MUL = 323,
     ASSIGN_SUB = 324,
     ASSIGN_ADD = 325,
     ARROW = 326,
     FOR_STEP = 327,
     OP_TO = 328,
     COMMA = 329,
     QUESTION = 330,
     OR = 331,
     AND = 332,
     NOT = 333,
     LET = 334,
     LE = 335,
     GE = 336,
     LT = 337,
     GT = 338,
     NEQ = 339,
     EEQ = 340,
     OP_EQ = 341,
     OP_ASSIGN = 342,
     PROVIDES = 343,
     OP_NOTIN = 344,
     OP_IN = 345,
     HASNT = 346,
     HAS = 347,
     DIESIS = 348,
     ATSIGN = 349,
     CAP = 350,
     VBAR = 351,
     AMPER = 352,
     MINUS = 353,
     PLUS = 354,
     PERCENT = 355,
     SLASH = 356,
     STAR = 357,
     POW = 358,
     SHR = 359,
     SHL = 360,
     BANG = 361,
     NEG = 362,
     DECREMENT = 363,
     INCREMENT = 364,
     DOLLAR = 365
   };
#endif
/* Tokens.  */
#define EOL 258
#define INTNUM 259
#define DBLNUM 260
#define SYMBOL 261
#define STRING 262
#define NIL 263
#define END 264
#define DEF 265
#define WHILE 266
#define BREAK 267
#define CONTINUE 268
#define DROPPING 269
#define IF 270
#define ELSE 271
#define ELIF 272
#define FOR 273
#define FORFIRST 274
#define FORLAST 275
#define FORMIDDLE 276
#define SWITCH 277
#define CASE 278
#define DEFAULT 279
#define SELECT 280
#define SENDER 281
#define SELF 282
#define GIVE 283
#define TRY 284
#define CATCH 285
#define RAISE 286
#define CLASS 287
#define FROM 288
#define OBJECT 289
#define RETURN 290
#define GLOBAL 291
#define LAMBDA 292
#define INIT 293
#define LOAD 294
#define LAUNCH 295
#define CONST_KW 296
#define ATTRIBUTES 297
#define PASS 298
#define EXPORT 299
#define DIRECTIVE 300
#define COLON 301
#define FUNCDECL 302
#define STATIC 303
#define FORDOT 304
#define LISTPAR 305
#define LOOP 306
#define TRUE_TOKEN 307
#define FALSE_TOKEN 308
#define OUTER_STRING 309
#define CLOSEPAR 310
#define OPENPAR 311
#define CLOSESQUARE 312
#define OPENSQUARE 313
#define DOT 314
#define ASSIGN_POW 315
#define ASSIGN_SHL 316
#define ASSIGN_SHR 317
#define ASSIGN_BXOR 318
#define ASSIGN_BOR 319
#define ASSIGN_BAND 320
#define ASSIGN_MOD 321
#define ASSIGN_DIV 322
#define ASSIGN_MUL 323
#define ASSIGN_SUB 324
#define ASSIGN_ADD 325
#define ARROW 326
#define FOR_STEP 327
#define OP_TO 328
#define COMMA 329
#define QUESTION 330
#define OR 331
#define AND 332
#define NOT 333
#define LET 334
#define LE 335
#define GE 336
#define LT 337
#define GT 338
#define NEQ 339
#define EEQ 340
#define OP_EQ 341
#define OP_ASSIGN 342
#define PROVIDES 343
#define OP_NOTIN 344
#define OP_IN 345
#define HASNT 346
#define HAS 347
#define DIESIS 348
#define ATSIGN 349
#define CAP 350
#define VBAR 351
#define AMPER 352
#define MINUS 353
#define PLUS 354
#define PERCENT 355
#define SLASH 356
#define STAR 357
#define POW 358
#define SHR 359
#define SHL 360
#define BANG 361
#define NEG 362
#define DECREMENT 363
#define INCREMENT 364
#define DOLLAR 365




/* Copy the first part of user declarations.  */
#line 22 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


#include <math.h>
#include <stdio.h>
#include <iostream>

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <stdlib.h>

#include <falcon/fassert.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::Compiler *>(yyparam) )
#define  CTX_LINE  ( COMPILER->lexer()->ctxOpenLine() )
#define  LINE      ( COMPILER->lexer()->previousLine() )
#define  CURRENT_LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM yyparam
#define YYLEX_PARAM yyparam

int flc_src_parse( void *param );
void flc_src_error (const char *s);

inline int flc_src_lex (void *lvalp, void *yyparam)
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
typedef union
#line 66 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
lex_value_t {
   Falcon::int64 integer;
   Falcon::numeric numeric;
   char * charp;
   Falcon::String *stringp;
   Falcon::Symbol *symbol;
   Falcon::Value *fal_val;
   Falcon::Expression *fal_exp;
   Falcon::Statement *fal_stat;
   Falcon::ArrayDecl *fal_adecl;
   Falcon::DictDecl *fal_ddecl;
   Falcon::SymbolList *fal_symlist;
}
/* Line 187 of yacc.c.  */
#line 380 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 393 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   6390

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  111
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  171
/* YYNRULES -- Number of rules.  */
#define YYNRULES  446
/* YYNRULES -- Number of states.  */
#define YYNSTATES  821

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   365

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
     105,   106,   107,   108,   109,   110
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    40,    44,
      48,    50,    52,    56,    60,    64,    67,    71,    77,    80,
      82,    84,    86,    88,    90,    92,    94,    96,    98,   100,
     102,   104,   106,   108,   110,   112,   114,   116,   118,   120,
     122,   126,   130,   135,   141,   146,   151,   158,   165,   167,
     169,   171,   173,   175,   177,   179,   181,   183,   185,   187,
     192,   197,   202,   207,   212,   217,   222,   227,   232,   237,
     242,   243,   249,   252,   256,   258,   262,   266,   269,   273,
     274,   281,   284,   288,   292,   296,   300,   301,   303,   304,
     308,   311,   315,   316,   321,   325,   329,   330,   333,   336,
     340,   343,   347,   351,   352,   358,   361,   369,   379,   383,
     391,   401,   405,   406,   416,   417,   425,   431,   432,   435,
     437,   439,   441,   443,   447,   451,   455,   458,   462,   465,
     469,   473,   475,   476,   483,   487,   491,   492,   499,   503,
     507,   508,   515,   519,   523,   524,   531,   535,   539,   540,
     543,   547,   549,   550,   556,   557,   563,   564,   570,   571,
     577,   578,   579,   583,   584,   586,   589,   592,   595,   597,
     601,   603,   605,   607,   611,   613,   614,   621,   625,   629,
     630,   633,   637,   639,   640,   646,   647,   653,   654,   660,
     661,   667,   669,   673,   674,   676,   678,   684,   689,   693,
     697,   698,   705,   708,   712,   713,   715,   717,   720,   723,
     726,   731,   735,   741,   745,   747,   751,   753,   755,   759,
     763,   769,   772,   778,   779,   787,   791,   797,   798,   805,
     808,   809,   811,   815,   817,   818,   819,   825,   826,   830,
     833,   837,   840,   844,   848,   852,   856,   862,   868,   872,
     878,   884,   888,   891,   895,   899,   901,   905,   909,   913,
     915,   919,   923,   927,   932,   936,   939,   943,   946,   950,
     951,   953,   957,   960,   964,   967,   968,   977,   981,   984,
     985,   989,   990,   996,   997,  1000,  1002,  1006,  1009,  1010,
    1014,  1016,  1020,  1022,  1024,  1026,  1027,  1030,  1032,  1034,
    1036,  1038,  1039,  1047,  1053,  1058,  1059,  1063,  1067,  1069,
    1072,  1076,  1081,  1082,  1091,  1094,  1097,  1098,  1101,  1103,
    1105,  1107,  1109,  1110,  1115,  1117,  1121,  1125,  1127,  1130,
    1134,  1138,  1140,  1142,  1144,  1146,  1148,  1150,  1152,  1154,
    1156,  1158,  1160,  1163,  1168,  1174,  1178,  1180,  1182,  1186,
    1189,  1193,  1197,  1201,  1205,  1209,  1213,  1217,  1221,  1225,
    1229,  1232,  1237,  1242,  1246,  1249,  1252,  1255,  1258,  1262,
    1266,  1270,  1274,  1278,  1282,  1286,  1290,  1294,  1297,  1301,
    1305,  1309,  1313,  1317,  1320,  1323,  1326,  1328,  1330,  1334,
    1339,  1345,  1348,  1350,  1352,  1354,  1356,  1360,  1364,  1369,
    1374,  1380,  1385,  1389,  1390,  1397,  1398,  1405,  1410,  1414,
    1417,  1418,  1424,  1426,  1429,  1435,  1441,  1446,  1450,  1453,
    1457,  1461,  1464,  1468,  1472,  1476,  1480,  1485,  1487,  1491,
    1493,  1496,  1498,  1502,  1504,  1508,  1512
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     112,     0,    -1,   113,    -1,    -1,   113,   114,    -1,   115,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   116,    -1,
     224,    -1,   206,    -1,   232,    -1,   250,    -1,   117,    -1,
     221,    -1,   222,    -1,   227,    -1,    39,     6,     3,    -1,
      39,     7,     3,    -1,    39,     1,     3,    -1,   119,    -1,
       3,    -1,    47,     1,     3,    -1,    34,     1,     3,    -1,
      32,     1,     3,    -1,     1,     3,    -1,   261,    87,   264,
      -1,   118,    74,   261,    87,   264,    -1,   264,     3,    -1,
     120,    -1,   121,    -1,   122,    -1,   134,    -1,   151,    -1,
     155,    -1,   169,    -1,   184,    -1,   138,    -1,   149,    -1,
     150,    -1,   195,    -1,   196,    -1,   205,    -1,   259,    -1,
     255,    -1,   219,    -1,   220,    -1,   160,    -1,   161,    -1,
     162,    -1,    10,   118,     3,    -1,    10,     1,     3,    -1,
     263,    87,   264,     3,    -1,   263,    87,   110,   110,     3,
      -1,   263,    87,   277,     3,    -1,   263,    87,   268,     3,
      -1,   263,    74,   280,    87,   264,     3,    -1,   263,    74,
     280,    87,   277,     3,    -1,   123,    -1,   124,    -1,   125,
      -1,   126,    -1,   127,    -1,   128,    -1,   129,    -1,   130,
      -1,   131,    -1,   132,    -1,   133,    -1,   264,    70,   264,
       3,    -1,   263,    69,   264,     3,    -1,   263,    68,   264,
       3,    -1,   263,    67,   264,     3,    -1,   263,    66,   264,
       3,    -1,   263,    60,   264,     3,    -1,   263,    65,   264,
       3,    -1,   263,    64,   264,     3,    -1,   263,    63,   264,
       3,    -1,   263,    61,   264,     3,    -1,   263,    62,   264,
       3,    -1,    -1,   136,   135,   148,     9,     3,    -1,   137,
     117,    -1,    11,   264,     3,    -1,    51,    -1,    11,     1,
       3,    -1,    11,   264,    46,    -1,    51,    46,    -1,    11,
       1,    46,    -1,    -1,   140,   139,   148,   142,     9,     3,
      -1,   141,   117,    -1,    15,   264,     3,    -1,    15,     1,
       3,    -1,    15,   264,    46,    -1,    15,     1,    46,    -1,
      -1,   145,    -1,    -1,   144,   143,   148,    -1,    16,     3,
      -1,    16,     1,     3,    -1,    -1,   147,   146,   148,   142,
      -1,    17,   264,     3,    -1,    17,     1,     3,    -1,    -1,
     148,   117,    -1,    12,     3,    -1,    12,     1,     3,    -1,
      13,     3,    -1,    13,    14,     3,    -1,    13,     1,     3,
      -1,    -1,   153,   152,   148,     9,     3,    -1,   154,   117,
      -1,    18,   263,    87,   264,    73,   264,     3,    -1,    18,
     263,    87,   264,    73,   264,    72,   264,     3,    -1,    18,
       1,     3,    -1,    18,   263,    87,   264,    73,   264,    46,
      -1,    18,   263,    87,   264,    73,   264,    72,   264,    46,
      -1,    18,     1,    46,    -1,    -1,    18,   279,    90,   264,
       3,   156,   158,     9,     3,    -1,    -1,    18,   279,    90,
     264,    46,   157,   117,    -1,    18,   279,    90,     1,     3,
      -1,    -1,   159,   158,    -1,   117,    -1,   163,    -1,   165,
      -1,   167,    -1,    49,   264,     3,    -1,    49,     1,     3,
      -1,   104,   277,     3,    -1,   104,     3,    -1,    83,   277,
       3,    -1,    83,     3,    -1,   104,     1,     3,    -1,    83,
       1,     3,    -1,    54,    -1,    -1,    19,     3,   164,   148,
       9,     3,    -1,    19,    46,   117,    -1,    19,     1,     3,
      -1,    -1,    20,     3,   166,   148,     9,     3,    -1,    20,
      46,   117,    -1,    20,     1,     3,    -1,    -1,    21,     3,
     168,   148,     9,     3,    -1,    21,    46,   117,    -1,    21,
       1,     3,    -1,    -1,   171,   170,   172,   178,     9,     3,
      -1,    22,   264,     3,    -1,    22,     1,     3,    -1,    -1,
     172,   173,    -1,   172,     1,     3,    -1,     3,    -1,    -1,
      23,   182,     3,   174,   148,    -1,    -1,    23,   182,    46,
     175,   117,    -1,    -1,    23,     1,     3,   176,   148,    -1,
      -1,    23,     1,    46,   177,   117,    -1,    -1,    -1,   180,
     179,   181,    -1,    -1,    24,    -1,    24,     1,    -1,     3,
     148,    -1,    46,   117,    -1,   183,    -1,   182,    74,   183,
      -1,     8,    -1,     4,    -1,     7,    -1,     4,    73,     4,
      -1,     6,    -1,    -1,   186,   185,   187,   178,     9,     3,
      -1,    25,   264,     3,    -1,    25,     1,     3,    -1,    -1,
     187,   188,    -1,   187,     1,     3,    -1,     3,    -1,    -1,
      23,   193,     3,   189,   148,    -1,    -1,    23,   193,    46,
     190,   117,    -1,    -1,    23,     1,     3,   191,   148,    -1,
      -1,    23,     1,    46,   192,   117,    -1,   194,    -1,   193,
      74,   194,    -1,    -1,     4,    -1,     6,    -1,    28,   277,
      73,   264,     3,    -1,    28,   277,     1,     3,    -1,    28,
       1,     3,    -1,    29,    46,   117,    -1,    -1,   198,   197,
     148,   199,     9,     3,    -1,    29,     3,    -1,    29,     1,
       3,    -1,    -1,   200,    -1,   201,    -1,   200,   201,    -1,
     202,   148,    -1,    30,     3,    -1,    30,    90,   261,     3,
      -1,    30,   203,     3,    -1,    30,   203,    90,   261,     3,
      -1,    30,     1,     3,    -1,   204,    -1,   203,    74,   204,
      -1,     4,    -1,     6,    -1,    31,   264,     3,    -1,    31,
       1,     3,    -1,   207,   214,   148,     9,     3,    -1,   209,
     117,    -1,   211,    56,   212,    55,     3,    -1,    -1,   211,
      56,   212,     1,   208,    55,     3,    -1,   211,     1,     3,
      -1,   211,    56,   212,    55,    46,    -1,    -1,   211,    56,
       1,   210,    55,    46,    -1,    47,     6,    -1,    -1,   213,
      -1,   212,    74,   213,    -1,     6,    -1,    -1,    -1,   217,
     215,   148,     9,     3,    -1,    -1,   218,   216,   117,    -1,
      48,     3,    -1,    48,     1,     3,    -1,    48,    46,    -1,
      48,     1,    46,    -1,    40,   266,     3,    -1,    40,     1,
       3,    -1,    43,   264,     3,    -1,    43,   264,    90,   264,
       3,    -1,    43,   264,    90,     1,     3,    -1,    43,     1,
       3,    -1,    41,     6,    87,   260,     3,    -1,    41,     6,
      87,     1,     3,    -1,    41,     1,     3,    -1,    44,     3,
      -1,    44,   223,     3,    -1,    44,     1,     3,    -1,     6,
      -1,   223,    74,     6,    -1,    45,   225,     3,    -1,    45,
       1,     3,    -1,   226,    -1,   225,    74,   226,    -1,     6,
      86,     6,    -1,     6,    86,     4,    -1,   228,   231,     9,
       3,    -1,   229,   230,     3,    -1,    42,     3,    -1,    42,
       1,     3,    -1,    42,    46,    -1,    42,     1,    46,    -1,
      -1,     6,    -1,   230,    74,     6,    -1,   230,     3,    -1,
     231,   230,     3,    -1,     1,     3,    -1,    -1,    32,     6,
     233,   234,   243,   248,     9,     3,    -1,   235,   237,     3,
      -1,     1,     3,    -1,    -1,    56,   212,    55,    -1,    -1,
      56,   212,     1,   236,    55,    -1,    -1,    33,   238,    -1,
     239,    -1,   238,    74,   239,    -1,     6,   240,    -1,    -1,
      56,   241,    55,    -1,   242,    -1,   241,    74,   242,    -1,
     260,    -1,     6,    -1,    27,    -1,    -1,   243,   244,    -1,
       3,    -1,   206,    -1,   247,    -1,   245,    -1,    -1,    38,
       3,   246,   214,   148,     9,     3,    -1,    48,     6,    87,
     264,     3,    -1,     6,    87,   264,     3,    -1,    -1,    92,
     249,     3,    -1,    92,     1,     3,    -1,     6,    -1,    78,
       6,    -1,   249,    74,     6,    -1,   249,    74,    78,     6,
      -1,    -1,    34,     6,   251,   252,   253,   248,     9,     3,
      -1,   237,     3,    -1,     1,     3,    -1,    -1,   253,   254,
      -1,     3,    -1,   206,    -1,   247,    -1,   245,    -1,    -1,
      36,   256,   257,     3,    -1,   258,    -1,   257,    74,   258,
      -1,   257,    74,     1,    -1,     6,    -1,    35,     3,    -1,
      35,   264,     3,    -1,    35,     1,     3,    -1,     8,    -1,
      52,    -1,    53,    -1,     4,    -1,     5,    -1,     7,    -1,
       6,    -1,   261,    -1,    27,    -1,    26,    -1,   262,    -1,
     263,   265,    -1,   263,    58,   264,    57,    -1,   263,    58,
     102,   264,    57,    -1,   263,    59,     6,    -1,   260,    -1,
     263,    -1,   264,    99,   264,    -1,    98,   264,    -1,   264,
      98,   264,    -1,   264,   102,   264,    -1,   264,   101,   264,
      -1,   264,   100,   264,    -1,   264,   103,   264,    -1,   264,
      97,   264,    -1,   264,    96,   264,    -1,   264,    95,   264,
      -1,   264,   105,   264,    -1,   264,   104,   264,    -1,   106,
     264,    -1,    79,   263,    86,   264,    -1,    79,   263,    87,
     264,    -1,   264,    84,   264,    -1,   264,   109,    -1,   109,
     264,    -1,   264,   108,    -1,   108,   264,    -1,   264,    85,
     264,    -1,   264,    86,   264,    -1,   264,    87,   264,    -1,
     264,    83,   264,    -1,   264,    82,   264,    -1,   264,    81,
     264,    -1,   264,    80,   264,    -1,   264,    77,   264,    -1,
     264,    76,   264,    -1,    78,   264,    -1,   264,    92,   264,
      -1,   264,    91,   264,    -1,   264,    90,   264,    -1,   264,
      89,   264,    -1,   264,    88,     6,    -1,   110,   264,    -1,
      94,   264,    -1,    93,   264,    -1,   271,    -1,   266,    -1,
     266,    59,     6,    -1,   266,    58,   264,    57,    -1,   266,
      58,   102,   264,    57,    -1,   266,   265,    -1,   274,    -1,
     275,    -1,   276,    -1,   265,    -1,    56,   264,    55,    -1,
      58,    46,    57,    -1,    58,   264,    46,    57,    -1,    58,
      46,   264,    57,    -1,    58,   264,    46,   264,    57,    -1,
     264,    56,   277,    55,    -1,   264,    56,    55,    -1,    -1,
     264,    56,   277,     1,   267,    55,    -1,    -1,    47,   269,
     270,   214,   148,     9,    -1,    56,   212,    55,     3,    -1,
      56,   212,     1,    -1,     1,     3,    -1,    -1,    37,   272,
     273,    71,   264,    -1,   212,    -1,     1,     3,    -1,   264,
      75,   264,    46,   264,    -1,   264,    75,   264,    46,     1,
      -1,   264,    75,   264,     1,    -1,   264,    75,     1,    -1,
      58,    57,    -1,    58,   277,    57,    -1,    58,   277,     1,
      -1,    50,    57,    -1,    50,   278,    57,    -1,    50,   278,
       1,    -1,    58,    71,    57,    -1,    58,   281,    57,    -1,
      58,   281,     1,    57,    -1,   264,    -1,   277,    74,   264,
      -1,   264,    -1,   278,   264,    -1,   261,    -1,   279,    74,
     261,    -1,   263,    -1,   280,    74,   263,    -1,   264,    71,
     264,    -1,   281,    74,   264,    71,   264,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   196,   196,   199,   201,   205,   206,   207,   212,   217,
     218,   223,   228,   233,   238,   239,   240,   244,   250,   256,
     264,   265,   268,   269,   270,   271,   276,   281,   288,   289,
     290,   291,   292,   293,   294,   295,   296,   297,   298,   299,
     300,   301,   302,   303,   304,   305,   306,   307,   308,   309,
     313,   315,   321,   325,   329,   333,   337,   342,   352,   353,
     354,   355,   356,   357,   358,   359,   360,   361,   362,   366,
     373,   380,   387,   394,   401,   408,   415,   422,   428,   434,
     442,   442,   456,   464,   465,   466,   470,   471,   472,   476,
     476,   491,   501,   502,   506,   507,   511,   513,   514,   514,
     523,   524,   529,   529,   541,   542,   545,   547,   553,   562,
     570,   580,   589,   597,   597,   611,   627,   631,   635,   643,
     647,   651,   661,   660,   685,   684,   710,   714,   716,   720,
     727,   728,   729,   733,   746,   754,   758,   764,   770,   777,
     782,   791,   801,   801,   815,   824,   828,   828,   841,   850,
     854,   854,   870,   879,   883,   883,   900,   901,   908,   910,
     911,   915,   917,   916,   927,   927,   939,   939,   951,   951,
     967,   970,   969,   982,   983,   984,   987,   988,   994,   995,
     999,  1008,  1020,  1031,  1042,  1063,  1063,  1080,  1081,  1088,
    1090,  1091,  1095,  1097,  1096,  1107,  1107,  1120,  1120,  1132,
    1132,  1150,  1151,  1154,  1155,  1167,  1188,  1192,  1197,  1205,
    1212,  1211,  1230,  1231,  1234,  1236,  1240,  1241,  1245,  1250,
    1268,  1288,  1298,  1309,  1317,  1318,  1322,  1334,  1357,  1358,
    1365,  1375,  1384,  1385,  1385,  1389,  1393,  1394,  1394,  1401,
    1455,  1457,  1458,  1462,  1477,  1480,  1479,  1491,  1490,  1505,
    1506,  1510,  1511,  1520,  1524,  1532,  1539,  1549,  1555,  1567,
    1577,  1582,  1594,  1603,  1610,  1618,  1623,  1635,  1640,  1648,
    1649,  1653,  1657,  1669,  1676,  1686,  1687,  1690,  1691,  1694,
    1696,  1700,  1707,  1708,  1709,  1721,  1720,  1779,  1782,  1788,
    1790,  1791,  1791,  1797,  1799,  1803,  1804,  1808,  1842,  1844,
    1853,  1854,  1858,  1859,  1868,  1871,  1873,  1877,  1878,  1881,
    1899,  1903,  1903,  1937,  1959,  1986,  1988,  1989,  1996,  2004,
    2010,  2016,  2030,  2029,  2093,  2094,  2100,  2102,  2106,  2107,
    2110,  2129,  2138,  2137,  2155,  2156,  2157,  2164,  2180,  2181,
    2182,  2192,  2193,  2194,  2195,  2196,  2197,  2201,  2219,  2220,
    2221,  2232,  2233,  2238,  2243,  2249,  2262,  2263,  2264,  2265,
    2266,  2267,  2268,  2269,  2270,  2271,  2272,  2273,  2274,  2275,
    2276,  2277,  2279,  2281,  2282,  2283,  2284,  2285,  2286,  2287,
    2288,  2289,  2290,  2291,  2292,  2293,  2294,  2295,  2296,  2297,
    2298,  2299,  2300,  2301,  2302,  2303,  2304,  2305,  2306,  2314,
    2318,  2322,  2326,  2327,  2328,  2329,  2330,  2335,  2338,  2341,
    2344,  2350,  2356,  2361,  2361,  2371,  2370,  2413,  2414,  2418,
    2427,  2426,  2470,  2471,  2480,  2485,  2492,  2499,  2509,  2510,
    2514,  2521,  2522,  2526,  2535,  2536,  2537,  2545,  2546,  2550,
    2551,  2555,  2561,  2568,  2574,  2581,  2582
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "END", "DEF", "WHILE", "BREAK", "CONTINUE", "DROPPING",
  "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST", "FORMIDDLE",
  "SWITCH", "CASE", "DEFAULT", "SELECT", "SENDER", "SELF", "GIVE", "TRY",
  "CATCH", "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL",
  "LAMBDA", "INIT", "LOAD", "LAUNCH", "CONST_KW", "ATTRIBUTES", "PASS",
  "EXPORT", "DIRECTIVE", "COLON", "FUNCDECL", "STATIC", "FORDOT",
  "LISTPAR", "LOOP", "TRUE_TOKEN", "FALSE_TOKEN", "OUTER_STRING",
  "CLOSEPAR", "OPENPAR", "CLOSESQUARE", "OPENSQUARE", "DOT", "ASSIGN_POW",
  "ASSIGN_SHL", "ASSIGN_SHR", "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND",
  "ASSIGN_MOD", "ASSIGN_DIV", "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD",
  "ARROW", "FOR_STEP", "OP_TO", "COMMA", "QUESTION", "OR", "AND", "NOT",
  "LET", "LE", "GE", "LT", "GT", "NEQ", "EEQ", "OP_EQ", "OP_ASSIGN",
  "PROVIDES", "OP_NOTIN", "OP_IN", "HASNT", "HAS", "DIESIS", "ATSIGN",
  "CAP", "VBAR", "AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR",
  "POW", "SHR", "SHL", "BANG", "NEG", "DECREMENT", "INCREMENT", "DOLLAR",
  "$accept", "input", "body", "line", "toplevel_statement",
  "load_statement", "statement", "assignment_def_list", "base_statement",
  "def_statement", "assignment", "op_assignment", "autoadd", "autosub",
  "automul", "autodiv", "automod", "autopow", "autoband", "autobor",
  "autobxor", "autoshl", "autoshr", "while_statement", "@1", "while_decl",
  "while_short_decl", "if_statement", "@2", "if_decl", "if_short_decl",
  "elif_or_else", "@3", "else_decl", "elif_statement", "@4", "elif_decl",
  "statement_list", "break_statement", "continue_statement",
  "for_statement", "@5", "for_decl", "for_decl_short", "forin_statement",
  "@6", "@7", "forin_statement_list", "forin_statement_elem",
  "fordot_statement", "self_print_statement", "outer_print_statement",
  "first_loop_block", "@8", "last_loop_block", "@9", "middle_loop_block",
  "@10", "switch_statement", "@11", "switch_decl", "case_list",
  "case_statement", "@12", "@13", "@14", "@15", "default_statement", "@16",
  "default_decl", "default_body", "case_expression_list", "case_element",
  "select_statement", "@17", "select_decl", "selcase_list",
  "selcase_statement", "@18", "@19", "@20", "@21",
  "selcase_expression_list", "selcase_element", "give_statement",
  "try_statement", "@22", "try_decl", "catch_statements", "catch_list",
  "catch_statement", "catch_decl", "catchcase_element_list",
  "catchcase_element", "raise_statement", "func_statement", "func_decl",
  "@23", "func_decl_short", "@24", "func_begin", "param_list",
  "param_symbol", "static_block", "@25", "@26", "static_decl",
  "static_short_decl", "launch_statement", "pass_statement",
  "const_statement", "export_statement", "export_symbol_list",
  "directive_statement", "directive_pair_list", "directive_pair",
  "attributes_statement", "attributes_decl", "attributes_short_decl",
  "attribute_list", "attribute_vert_list", "class_decl", "@27",
  "class_def_inner", "class_param_list", "@28", "from_clause",
  "inherit_list", "inherit_token", "inherit_call", "inherit_param_list",
  "inherit_param_token", "class_statement_list", "class_statement",
  "init_decl", "@29", "property_decl", "has_list", "has_clause_list",
  "object_decl", "@30", "object_decl_inner", "object_statement_list",
  "object_statement", "global_statement", "@31", "global_symbol_list",
  "globalized_symbol", "return_statement", "const_atom", "atomic_symbol",
  "var_atom", "variable", "expression", "range_decl", "func_call", "@32",
  "nameless_func", "@33", "nameless_func_decl_inner", "lambda_expr", "@34",
  "lambda_expr_inner", "iif_expr", "array_decl", "dict_decl",
  "expression_list", "listpar_expression_list", "symbol_list",
  "assignment_list", "expression_pair_list", 0
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
     365
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   111,   112,   113,   113,   114,   114,   114,   115,   115,
     115,   115,   115,   115,   115,   115,   115,   116,   116,   116,
     117,   117,   117,   117,   117,   117,   118,   118,   119,   119,
     119,   119,   119,   119,   119,   119,   119,   119,   119,   119,
     119,   119,   119,   119,   119,   119,   119,   119,   119,   119,
     120,   120,   121,   121,   121,   121,   121,   121,   122,   122,
     122,   122,   122,   122,   122,   122,   122,   122,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     135,   134,   134,   136,   136,   136,   137,   137,   137,   139,
     138,   138,   140,   140,   141,   141,   142,   142,   143,   142,
     144,   144,   146,   145,   147,   147,   148,   148,   149,   149,
     150,   150,   150,   152,   151,   151,   153,   153,   153,   154,
     154,   154,   156,   155,   157,   155,   155,   158,   158,   159,
     159,   159,   159,   160,   160,   161,   161,   161,   161,   161,
     161,   162,   164,   163,   163,   163,   166,   165,   165,   165,
     168,   167,   167,   167,   170,   169,   171,   171,   172,   172,
     172,   173,   174,   173,   175,   173,   176,   173,   177,   173,
     178,   179,   178,   180,   180,   180,   181,   181,   182,   182,
     183,   183,   183,   183,   183,   185,   184,   186,   186,   187,
     187,   187,   188,   189,   188,   190,   188,   191,   188,   192,
     188,   193,   193,   194,   194,   194,   195,   195,   195,   196,
     197,   196,   198,   198,   199,   199,   200,   200,   201,   202,
     202,   202,   202,   202,   203,   203,   204,   204,   205,   205,
     206,   206,   207,   208,   207,   207,   209,   210,   209,   211,
     212,   212,   212,   213,   214,   215,   214,   216,   214,   217,
     217,   218,   218,   219,   219,   220,   220,   220,   220,   221,
     221,   221,   222,   222,   222,   223,   223,   224,   224,   225,
     225,   226,   226,   227,   227,   228,   228,   229,   229,   230,
     230,   230,   231,   231,   231,   233,   232,   234,   234,   235,
     235,   236,   235,   237,   237,   238,   238,   239,   240,   240,
     241,   241,   242,   242,   242,   243,   243,   244,   244,   244,
     244,   246,   245,   247,   247,   248,   248,   248,   249,   249,
     249,   249,   251,   250,   252,   252,   253,   253,   254,   254,
     254,   254,   256,   255,   257,   257,   257,   258,   259,   259,
     259,   260,   260,   260,   260,   260,   260,   261,   262,   262,
     262,   263,   263,   263,   263,   263,   264,   264,   264,   264,
     264,   264,   264,   264,   264,   264,   264,   264,   264,   264,
     264,   264,   264,   264,   264,   264,   264,   264,   264,   264,
     264,   264,   264,   264,   264,   264,   264,   264,   264,   264,
     264,   264,   264,   264,   264,   264,   264,   264,   264,   264,
     264,   264,   264,   264,   264,   264,   264,   265,   265,   265,
     265,   266,   266,   267,   266,   269,   268,   270,   270,   270,
     272,   271,   273,   273,   274,   274,   274,   274,   275,   275,
     275,   275,   275,   275,   276,   276,   276,   277,   277,   278,
     278,   279,   279,   280,   280,   281,   281
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     3,     3,
       1,     1,     3,     3,     3,     2,     3,     5,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       3,     3,     4,     5,     4,     4,     6,     6,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       0,     5,     2,     3,     1,     3,     3,     2,     3,     0,
       6,     2,     3,     3,     3,     3,     0,     1,     0,     3,
       2,     3,     0,     4,     3,     3,     0,     2,     2,     3,
       2,     3,     3,     0,     5,     2,     7,     9,     3,     7,
       9,     3,     0,     9,     0,     7,     5,     0,     2,     1,
       1,     1,     1,     3,     3,     3,     2,     3,     2,     3,
       3,     1,     0,     6,     3,     3,     0,     6,     3,     3,
       0,     6,     3,     3,     0,     6,     3,     3,     0,     2,
       3,     1,     0,     5,     0,     5,     0,     5,     0,     5,
       0,     0,     3,     0,     1,     2,     2,     2,     1,     3,
       1,     1,     1,     3,     1,     0,     6,     3,     3,     0,
       2,     3,     1,     0,     5,     0,     5,     0,     5,     0,
       5,     1,     3,     0,     1,     1,     5,     4,     3,     3,
       0,     6,     2,     3,     0,     1,     1,     2,     2,     2,
       4,     3,     5,     3,     1,     3,     1,     1,     3,     3,
       5,     2,     5,     0,     7,     3,     5,     0,     6,     2,
       0,     1,     3,     1,     0,     0,     5,     0,     3,     2,
       3,     2,     3,     3,     3,     3,     5,     5,     3,     5,
       5,     3,     2,     3,     3,     1,     3,     3,     3,     1,
       3,     3,     3,     4,     3,     2,     3,     2,     3,     0,
       1,     3,     2,     3,     2,     0,     8,     3,     2,     0,
       3,     0,     5,     0,     2,     1,     3,     2,     0,     3,
       1,     3,     1,     1,     1,     0,     2,     1,     1,     1,
       1,     0,     7,     5,     4,     0,     3,     3,     1,     2,
       3,     4,     0,     8,     2,     2,     0,     2,     1,     1,
       1,     1,     0,     4,     1,     3,     3,     1,     2,     3,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     4,     5,     3,     1,     1,     3,     2,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       2,     4,     4,     3,     2,     2,     2,     2,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     2,     3,     3,
       3,     3,     3,     2,     2,     2,     1,     1,     3,     4,
       5,     2,     1,     1,     1,     1,     3,     3,     4,     4,
       5,     4,     3,     0,     6,     0,     6,     4,     3,     2,
       0,     5,     1,     2,     5,     5,     4,     3,     2,     3,
       3,     2,     3,     3,     3,     3,     4,     1,     3,     1,
       2,     1,     3,     1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    21,   344,   345,   347,   346,
     341,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   350,   349,     0,     0,     0,     0,     0,     0,   332,
     420,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    84,   342,   343,   141,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     4,     5,
       8,    13,    20,    29,    30,    31,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    32,    80,     0,
      37,    89,     0,    38,    39,    33,   113,     0,    34,    47,
      48,    49,    35,   154,    36,   185,    40,    41,   210,    42,
      10,   244,     0,     0,    45,    46,    14,    15,     9,    16,
       0,   279,    11,    12,    44,    43,   356,   348,   351,   357,
       0,   405,   397,   396,   402,   403,   404,    25,     6,     0,
       0,     0,     0,   357,     0,     0,   108,     0,   110,     0,
       0,     0,     0,   348,     0,     0,     0,     0,     0,     0,
       0,     0,   437,     0,     0,   212,     0,     0,     0,     0,
     285,     0,   322,     0,   338,     0,     0,     0,     0,     0,
       0,     0,     0,   397,     0,     0,     0,   275,   277,     0,
       0,     0,   262,   265,     0,     0,     0,     0,   269,     0,
     239,     0,     0,   431,   439,     0,    87,     0,     0,   428,
       0,   437,     0,     0,   387,     0,     0,   138,     0,   395,
     394,   359,     0,   136,     0,   370,   377,   375,   393,   106,
       0,     0,     0,    82,   106,    91,   106,   115,   158,   189,
     106,     0,   106,   245,   247,   231,     0,     0,     0,   280,
       0,   279,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   352,    28,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   376,   374,
       0,     0,   401,    51,    50,     0,     0,    85,    88,    83,
      86,   109,   112,   111,    93,    95,    92,    94,   118,   121,
       0,     0,     0,   157,   156,     7,   188,   187,   208,     0,
       0,     0,   213,   209,   229,   228,    24,     0,    23,     0,
     340,   339,   337,     0,   334,     0,   243,   422,   241,     0,
      19,    17,    18,   254,   253,   261,     0,   276,   278,   258,
     255,     0,   264,   263,     0,   268,     0,   267,     0,    22,
     134,   133,   433,   432,   440,   406,   407,     0,   434,     0,
       0,   430,   429,     0,   435,     0,     0,     0,   140,   137,
     139,   135,     0,     0,     0,     0,     0,     0,     0,   249,
     251,     0,   106,     0,   235,   237,     0,   284,   282,     0,
       0,     0,   274,     0,     0,   355,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   443,     0,   415,     0,
     437,     0,     0,   412,     0,     0,   427,     0,   386,   385,
     384,   383,   382,   381,   373,   378,   379,   380,   392,   391,
     390,   389,   388,   367,   366,   365,   360,   358,   363,   362,
     361,   364,   369,   368,     0,     0,   398,     0,    26,     0,
     442,     0,     0,   207,     0,   438,     0,   240,   305,   293,
       0,     0,     0,   326,   333,     0,   423,     0,     0,     0,
       0,     0,   390,   266,   272,   271,   270,   409,   408,     0,
     445,   436,     0,   371,   372,     0,   107,     0,     0,     0,
      98,    97,   102,     0,     0,   161,     0,     0,   159,     0,
     171,     0,   192,     0,     0,   190,     0,     0,   215,   216,
     106,   250,   252,     0,     0,   248,     0,   233,     0,   281,
     273,   283,     0,   353,    74,    78,    79,    77,    76,    75,
      73,    72,    71,    70,     0,     0,     0,     0,    52,    55,
      54,   413,   411,    69,   426,     0,     0,   399,     0,     0,
     126,   122,   124,   206,   288,     0,   315,     0,   325,   298,
     294,   295,   324,   315,   336,   335,   242,   421,   260,   259,
     257,   256,   410,     0,    81,     0,   100,     0,     0,     0,
     106,   106,   114,   160,     0,   181,   184,   182,   180,     0,
     178,   175,     0,     0,   191,     0,   204,   205,     0,   201,
       0,     0,   219,   226,   227,     0,     0,   224,     0,   217,
       0,   230,     0,     0,     0,   232,   236,   354,   444,   437,
       0,     0,   240,   244,    53,     0,   425,   424,   400,    27,
       0,     0,     0,   291,   290,   307,     0,     0,     0,     0,
       0,   308,   306,   310,   309,     0,   287,     0,   297,     0,
     328,   329,   331,   330,     0,   327,   446,   101,   105,   104,
      90,     0,     0,   166,   168,     0,   162,   164,     0,   155,
     106,     0,   172,   197,   199,   193,   195,   203,   186,   223,
       0,   221,     0,     0,   211,   246,   238,     0,    56,    57,
     419,     0,   106,   414,   116,   119,     0,     0,     0,     0,
     129,     0,     0,   130,   131,   132,   125,     0,     0,   311,
       0,     0,   318,     0,     0,     0,   303,   304,     0,   300,
     302,   296,     0,   103,   106,     0,   183,   106,     0,   179,
       0,   177,   106,     0,   106,     0,   202,   220,   225,     0,
     234,   418,     0,     0,     0,     0,   142,     0,     0,   146,
       0,     0,   150,     0,     0,   128,   292,     0,   244,     0,
     317,   319,   316,     0,   286,   299,     0,   323,     0,   169,
       0,   165,     0,   200,     0,   196,   222,   417,   416,   117,
     120,   145,   106,   144,   149,   106,   148,   153,   106,   152,
     123,   314,   106,     0,   320,     0,   301,     0,     0,     0,
       0,   313,   321,     0,     0,     0,     0,   143,   147,   151,
     312
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    58,    59,    60,   496,   130,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,   219,    78,    79,    80,   224,    81,
      82,   499,   590,   500,   501,   591,   502,   382,    83,    84,
      85,   226,    86,    87,    88,   641,   642,   711,   712,    89,
      90,    91,   713,   792,   714,   795,   715,   798,    92,   228,
      93,   385,   508,   737,   738,   734,   735,   509,   603,   510,
     682,   599,   600,    94,   229,    95,   386,   515,   744,   745,
     742,   743,   608,   609,    96,    97,   230,    98,   517,   518,
     519,   520,   616,   617,    99,   100,   101,   624,   102,   526,
     103,   337,   338,   232,   392,   393,   233,   234,   104,   105,
     106,   107,   184,   108,   187,   188,   109,   110,   111,   240,
     241,   112,   327,   468,   469,   717,   472,   570,   571,   658,
     728,   729,   566,   652,   653,   768,   654,   655,   724,   113,
     329,   473,   573,   665,   114,   166,   333,   334,   115,   116,
     117,   118,   133,   120,   121,   122,   635,   421,   546,   633,
     123,   167,   339,   124,   125,   126,   153,   195,   145,   417,
     203
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -631
static const yytype_int16 yypact[] =
{
    -631,    61,   809,  -631,     5,  -631,  -631,  -631,  -631,  -631,
    -631,    35,    49,  3133,   222,   427,  3167,   338,  3229,    86,
    3263,  -631,  -631,  3325,   183,  3359,   314,   354,    87,  -631,
    -631,   243,  3421,   442,   202,  3455,   340,   481,   513,  3517,
    5402,    55,  -631,  -631,  -631,  5642,  5215,  5642,   411,   346,
    5642,  5642,  5642,   543,  5642,  5642,  5642,  5642,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  3009,
    -631,  -631,  3009,  -631,  -631,  -631,  -631,  3009,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,    63,  3009,   106,  -631,  -631,  -631,  -631,  -631,  -631,
      38,   186,  -631,  -631,  -631,  -631,  -631,  -631,  -631,   630,
    3978,  -631,   459,  -631,  -631,  -631,  -631,  -631,  -631,   174,
      48,    66,   164,   479,  4038,   219,  -631,   272,  -631,   308,
     165,  4098,   184,   171,   364,   352,   320,  4268,   331,   339,
    4322,   342,  6170,    78,   345,  -631,  3009,   381,  4359,   397,
    -631,   418,  -631,   431,  -631,  4397,   250,    52,   443,   450,
     464,   467,  6170,   141,   480,   280,   185,  -631,  -631,   483,
    4434,   491,  -631,  -631,    51,   504,   447,    95,  -631,   538,
    -631,   540,  4472,  -631,  6170,  3071,  -631,  5898,  5436,  -631,
     488,  5740,     8,   102,  6281,   453,   555,  -631,   130,   310,
     310,   -49,   556,  -631,   133,   -49,   -49,   -49,   -49,  -631,
     565,   566,   567,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,   271,  -631,  -631,  -631,  -631,   570,    42,   571,  -631,
     139,    14,   144,  5291,   573,  5642,  5642,  5642,  5642,  5642,
    5642,  5642,  5642,  5642,  5642,   411,  5512,  -631,  -631,  5546,
    5642,  3551,  5642,  5642,  5642,  5642,  5642,  5642,  5642,  5642,
    5642,  5642,   576,  5642,  5642,  5642,  5642,  5642,  5642,  5642,
    5642,  5642,  5642,  5642,  5642,  5642,  5642,  5642,  -631,  -631,
    5325,   577,  -631,  -631,  -631,   586,  5642,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    5642,   586,  3613,  -631,  -631,  -631,  -631,  -631,  -631,   574,
    5642,  5642,  -631,  -631,  -631,  -631,  -631,   307,  -631,   400,
    -631,  -631,  -631,   145,  -631,   591,  -631,   523,  -631,   527,
    -631,  -631,  -631,  -631,  -631,  -631,   317,  -631,  -631,  -631,
    -631,  3647,  -631,  -631,   596,  -631,   384,  -631,   597,  -631,
    -631,  -631,  -631,  -631,  6170,  -631,  -631,  5935,  -631,  5608,
    5642,  -631,  -631,   547,  -631,  5642,  5642,  5642,  -631,  -631,
    -631,  -631,  1799,  1469,  1909,   426,   562,  1579,   205,  -631,
    -631,  2019,  -631,  3009,  -631,  -631,   101,  -631,  -631,   599,
     603,   146,  -631,  5642,  5800,  -631,  4509,  4547,  4584,  4622,
    4659,  4697,  4734,  4772,  4809,  4847,   479,    96,  -631,  5704,
    4884,   604,   147,  -631,   105,  4922,  -631,  3782,  6244,  6281,
     401,   401,   401,   401,   401,   401,   401,   401,  -631,   310,
     310,   310,   310,   376,   376,   360,   452,   452,   253,   253,
     253,   150,   -49,   -49,  5642,  5860,  -631,   529,  6170,  5972,
    -631,   612,  4158,  -631,  4959,  6170,   614,   613,  -631,   587,
     620,   618,   622,  -631,  -631,   515,  -631,   613,  5642,   623,
     624,   625,   138,  -631,  -631,  -631,  -631,  -631,  -631,  6009,
    6170,  -631,  6059,   401,   401,   626,  -631,   444,  3709,   621,
    -631,  -631,  -631,   628,   629,  -631,   528,   313,  -631,   633,
    -631,   631,  -631,   128,   636,  -631,    25,   637,   605,  -631,
    -631,  -631,  -631,   635,  2129,  -631,   592,  -631,   206,  -631,
    -631,  -631,  6096,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,   411,  5642,   160,  3889,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  3743,  6133,  -631,  5642,  5642,
    -631,  -631,  -631,  -631,  -631,   108,    72,   645,  -631,   594,
     580,  -631,  -631,    80,  -631,  -631,  -631,  6170,  -631,  -631,
    -631,  -631,  -631,  5642,  -631,   652,  -631,   653,  4997,   654,
    -631,  -631,  -631,  -631,   211,   588,  -631,  -631,  -631,   127,
    -631,  -631,   657,   328,  -631,   333,  -631,  -631,   143,  -631,
     659,   660,  -631,  -631,  -631,   586,    31,  -631,   661,  -631,
    1689,  -631,   662,   627,   611,  -631,  -631,  -631,   479,  5034,
     152,   664,   613,    63,  -631,   616,  -631,  6207,  -631,  6170,
    3928,   919,  3009,  -631,  -631,  -631,   581,   669,   668,   670,
      18,  -631,  -631,  -631,  -631,   666,  -631,   606,  -631,   618,
    -631,  -631,  -631,  -631,   671,  -631,  6170,  -631,  -631,  -631,
    -631,  2239,  1469,  -631,  -631,   673,  -631,  -631,   583,  -631,
    -631,  3009,  -631,  -631,  -631,  -631,  -631,   520,  -631,  -631,
     675,  -631,   521,   586,  -631,  -631,  -631,   678,  -631,  -631,
    -631,   116,  -631,  -631,  -631,  -631,  5642,   325,   329,   332,
    -631,   676,   919,  -631,  -631,  -631,  -631,   646,  5642,  -631,
     595,   681,  -631,   680,   154,   697,  -631,  -631,   -42,  -631,
    -631,  -631,   699,  -631,  -631,  3009,  -631,  -631,  3009,  -631,
    2349,  -631,  -631,  3009,  -631,  3009,  -631,  -631,  -631,   700,
    -631,  -631,   702,  2459,  4218,   703,  -631,  3009,   704,  -631,
    3009,   705,  -631,  3009,   706,  -631,  -631,  5072,    63,  5642,
    -631,  -631,  -631,    21,  -631,  -631,   606,  -631,  1029,  -631,
    1139,  -631,  1249,  -631,  1359,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  5109,  -631,   707,  -631,  2569,  2679,  2789,
    2899,  -631,  -631,   708,   709,   712,   713,  -631,  -631,  -631,
    -631
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -631,  -631,  -631,  -631,  -631,  -631,     2,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,    46,  -631,  -631,  -631,  -631,  -631,  -214,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,     7,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,   334,  -631,  -631,
    -631,  -631,    43,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,    37,  -631,  -631,  -631,  -631,  -631,  -631,
     204,  -631,  -631,    33,  -631,  -375,  -631,  -631,  -631,  -631,
    -631,  -235,   249,  -630,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,   369,  -631,  -631,  -631,  -106,
    -631,  -631,  -631,  -631,  -631,  -631,   259,  -631,    70,  -631,
    -631,   -46,  -631,  -631,   158,  -631,   159,   161,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,   264,  -631,  -345,
      -6,  -631,    -2,    17,   -97,   710,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,   -32,  -631,  -631,  -631,
    -631
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -442
static const yytype_int16 yytable[] =
{
     119,   480,   396,   702,    61,   242,   131,   259,   127,   371,
     383,   143,   384,   775,   202,   144,   387,   208,   391,   721,
     239,   214,   257,   400,   722,   292,   611,   804,   612,   613,
     134,   614,   776,   141,   691,   147,   257,   150,   128,   238,
     152,  -279,   158,   395,   239,   165,   205,   257,   336,   172,
     129,   294,   180,   335,   353,     8,   192,   194,   336,   288,
     289,     3,   197,   201,   204,   372,   152,   209,   210,   211,
     152,   215,   216,   217,   218,   645,   292,   119,   646,   319,
     119,   223,   321,   660,   225,   119,   646,   148,   163,   227,
     164,     6,     7,     8,     9,    10,   723,  -240,   357,   805,
     119,   196,   527,   373,   235,   692,   551,   236,   257,   643,
     647,   231,  -279,    21,    22,   615,  -240,   751,   647,   648,
     649,   693,   295,  -240,    30,   354,  -240,   648,   649,   605,
     676,  -203,   606,   379,   607,   401,   381,    40,   802,    42,
      43,   581,   398,    45,   344,    46,   685,   402,   474,   531,
     550,   320,   321,   296,   119,   699,   528,   772,   323,   374,
     552,   631,   237,   644,   650,    47,    48,   297,   304,   358,
     544,   752,   650,   677,  -203,   477,   375,   293,   524,   321,
      50,    51,   477,   545,   154,    52,   155,   308,   347,   686,
     477,   651,   239,    54,   259,    55,    56,    57,   661,   290,
     291,   678,  -203,   176,   321,   177,   259,   321,   521,   625,
     298,   305,   364,   399,   673,   367,   632,   687,   399,   475,
     399,   321,   301,   135,   422,   136,   321,   424,   773,   156,
     309,   348,   565,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   168,  -441,   288,   289,   178,   169,
     170,   522,   626,   416,   286,   287,   332,   674,   288,   289,
     404,  -441,   406,   407,   408,   409,   410,   411,   412,   413,
     414,   415,   388,   420,   389,   302,   152,   425,   427,   428,
     429,   430,   431,   432,   433,   434,   435,   436,   437,   457,
     439,   440,   441,   442,   443,   444,   445,   446,   447,   448,
     449,   450,   451,   452,   453,   460,   620,   455,   466,   259,
    -289,   303,   730,   458,   601,   159,  -174,   390,   479,   257,
     160,     6,     7,   313,     9,    10,   755,   459,   756,   462,
     758,   680,   759,   761,   315,   762,   683,   464,   465,   142,
    -289,   181,   316,   182,     8,   318,   183,   206,   322,   207,
       6,     7,     8,     9,    10,   161,   285,   286,   287,  -174,
     162,   288,   289,   467,    21,    22,   259,   346,   482,    42,
      43,   757,    21,    22,   681,   760,   671,   672,   763,   684,
     119,   119,   119,    30,   324,   119,   489,   490,   484,   119,
     485,   119,   492,   493,   494,   525,    40,   701,    42,    43,
     326,   470,    45,  -293,    46,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   286,   287,   259,     8,   288,   289,
     532,   328,   243,   244,    47,    48,   311,   504,   137,   505,
     138,   730,   259,   471,   330,  -170,   218,    21,    22,    50,
      51,   139,   312,   174,    52,   585,   340,   586,   175,   506,
     507,   310,    54,   341,    55,    56,    57,   259,   280,   281,
     282,   283,   284,   285,   286,   287,   740,   342,   288,   289,
     343,   556,  -173,   279,   280,   281,   282,   283,   284,   285,
     286,   287,   185,   345,   288,   289,   349,   186,   753,   272,
     273,   274,   275,   276,   352,   577,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   355,   259,   288,
     289,   243,   244,   630,   189,   588,   574,   290,   291,   190,
     778,   332,   119,   780,   606,   613,   607,   614,   782,   594,
     784,   257,   595,   356,   596,   597,   598,   243,   244,   376,
     377,   359,   628,   360,   212,   368,   213,     6,     7,     8,
       9,    10,   282,   283,   284,   285,   286,   287,   378,   380,
     288,   289,   629,   511,   218,   512,   159,   161,   189,    21,
      22,  -170,   637,   394,   397,   639,   640,   463,   807,   405,
      30,   808,   438,   456,   809,   513,   507,   595,   810,   596,
     597,   598,     8,    40,   476,    42,    43,   477,   478,    45,
     666,    46,   483,   186,   491,   529,   530,   549,  -173,   690,
       6,     7,   726,     9,    10,   560,   558,   564,   119,   336,
     471,    47,    48,   568,   569,   572,   578,   579,   580,   584,
     589,   592,   593,   727,   604,   516,    50,    51,   621,   119,
     119,    52,   602,   710,   716,   610,   618,   623,   656,    54,
     657,    55,    56,    57,   659,   667,   668,   670,    42,    43,
     679,   675,   688,   689,   694,   695,   697,   700,   718,   119,
     119,   703,   719,   696,   190,   725,   720,   736,   747,   119,
     732,   750,   769,   741,   770,   764,   771,   749,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     774,   766,   777,   786,   255,   787,   791,   794,   797,   800,
     119,   817,   818,   812,   710,   819,   820,   256,   733,   765,
     514,   739,   619,   754,   746,   748,   576,   486,   567,   731,
     806,   662,   663,   119,   664,   767,   119,   779,   119,   575,
     781,   119,   173,   119,     0,   783,     0,   785,     0,     0,
       0,   119,     0,     0,     0,   119,     0,     0,   119,   793,
       0,   119,   796,     0,     0,   799,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   119,     0,   119,     0,
     119,     0,   119,     0,     0,     0,   803,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   119,   119,   119,   119,    -2,
       4,     0,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,    19,     0,    20,    21,    22,    23,    24,     0,
      25,    26,     0,    27,    28,    29,    30,     0,    31,    32,
      33,    34,    35,    36,    37,     0,    38,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,  -127,    12,
      13,    14,    15,     0,    16,     0,     0,    17,   707,   708,
     709,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,  -167,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,  -167,  -167,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,  -167,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,  -163,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,  -163,  -163,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,  -163,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,  -198,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,  -198,  -198,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,  -198,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,  -194,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,  -194,  -194,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,  -194,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   -96,    12,
      13,    14,    15,     0,    16,   497,   498,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,  -214,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,   516,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,  -218,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,  -218,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   495,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   503,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   523,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   622,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   -99,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,  -176,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   788,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   813,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   814,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   815,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,   816,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       4,     0,     5,     6,     7,     8,     9,    10,     0,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   220,     0,   221,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   222,     0,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,     0,     0,
       0,     0,   362,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,    49,     0,     0,     0,     0,    21,    22,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    30,     0,
       0,     0,     0,    53,     0,    54,     0,    55,    56,    57,
       0,    40,     0,    42,    43,     0,     0,    45,   363,    46,
       0,     0,     0,     0,   132,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,    47,
      48,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,    50,    51,     0,     0,   140,    52,
      30,     6,     7,     8,     9,    10,     0,    54,     0,    55,
      56,    57,     0,    40,     0,    42,    43,     0,     0,    45,
       0,    46,     0,    21,    22,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    30,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,     0,    40,     0,    42,
      43,     0,     0,    45,     0,    46,    50,    51,     0,     0,
     146,    52,     0,     6,     7,     8,     9,    10,     0,    54,
       0,    55,    56,    57,     0,    47,    48,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
      50,    51,     0,     0,   149,    52,    30,     6,     7,     8,
       9,    10,     0,    54,     0,    55,    56,    57,     0,    40,
       0,    42,    43,     0,     0,    45,     0,    46,     0,    21,
      22,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      30,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,     0,    40,     0,    42,    43,     0,     0,    45,
       0,    46,    50,    51,     0,     0,   151,    52,     0,     6,
       7,     8,     9,    10,     0,    54,     0,    55,    56,    57,
       0,    47,    48,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,    50,    51,     0,     0,
     157,    52,    30,     6,     7,     8,     9,    10,     0,    54,
       0,    55,    56,    57,     0,    40,     0,    42,    43,     0,
       0,    45,     0,    46,     0,    21,    22,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    30,     0,     0,     0,
       0,     0,     0,    47,    48,     0,     0,     0,     0,    40,
       0,    42,    43,     0,     0,    45,     0,    46,    50,    51,
       0,     0,   171,    52,     0,     6,     7,     8,     9,    10,
       0,    54,     0,    55,    56,    57,     0,    47,    48,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,     0,    50,    51,     0,     0,   179,    52,    30,     6,
       7,     8,     9,    10,     0,    54,     0,    55,    56,    57,
       0,    40,     0,    42,    43,     0,     0,    45,     0,    46,
       0,    21,    22,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,     0,    47,
      48,     0,     0,     0,     0,    40,     0,    42,    43,     0,
       0,    45,     0,    46,    50,    51,     0,     0,   191,    52,
       0,     6,     7,     8,     9,    10,     0,    54,     0,    55,
      56,    57,     0,    47,    48,     0,     0,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,    50,    51,
       0,     0,   426,    52,    30,     6,     7,     8,     9,    10,
       0,    54,     0,    55,    56,    57,     0,    40,     0,    42,
      43,     0,     0,    45,     0,    46,     0,    21,    22,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    30,     0,
       0,     0,     0,     0,     0,    47,    48,     0,     0,     0,
       0,    40,     0,    42,    43,     0,     0,    45,     0,    46,
      50,    51,     0,     0,   461,    52,     0,     6,     7,     8,
       9,    10,     0,    54,     0,    55,    56,    57,     0,    47,
      48,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,    50,    51,     0,     0,   481,    52,
      30,     6,     7,     8,     9,    10,     0,    54,     0,    55,
      56,    57,     0,    40,     0,    42,    43,     0,     0,    45,
       0,    46,     0,    21,    22,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    30,     0,     0,     0,     0,     0,
       0,    47,    48,     0,     0,     0,     0,    40,     0,    42,
      43,     0,     0,    45,     0,    46,    50,    51,     0,     0,
     587,    52,     0,     6,     7,     8,     9,    10,     0,    54,
       0,    55,    56,    57,     0,    47,    48,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
      50,    51,     0,     0,   636,    52,    30,     6,     7,     8,
       9,    10,     0,    54,     0,    55,    56,    57,     0,    40,
       0,    42,    43,     0,     0,    45,     0,    46,     0,    21,
      22,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      30,     0,     0,   554,     0,     0,     0,    47,    48,     0,
       0,     0,     0,    40,     0,    42,    43,     0,     0,    45,
       0,    46,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,     0,     0,    54,     0,    55,    56,    57,
       0,    47,    48,     0,     0,     0,     0,     0,   555,     0,
       0,     0,     0,     0,     0,     0,    50,    51,   259,     0,
       0,    52,     0,     0,     0,     0,     0,     0,     0,    54,
       0,    55,    56,    57,     0,     0,     0,   261,   262,   263,
       0,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,     0,     0,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,     0,     0,
     288,   289,   634,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    30,     0,     0,     0,
       0,   704,     0,     0,     0,     0,     0,     0,     0,    40,
       0,    42,    43,     0,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     0,     0,     0,   705,     0,     0,     0,     0,     0,
       0,   258,    50,    51,   259,     0,     0,    52,     0,     0,
       0,     0,     0,     0,     0,    54,     0,    55,    56,    57,
     706,     0,     0,   261,   262,   263,     0,     0,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,     0,     0,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   259,     0,   288,   289,     0,     0,
       0,   299,     0,     0,     0,     0,     0,     0,   260,     0,
       0,     0,     0,   261,   262,   263,     0,     0,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,     0,     0,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   300,     0,   288,   289,     0,     0,
       0,     0,     0,     0,   259,     0,     0,     0,     0,     0,
       0,   306,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   261,   262,   263,     0,     0,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,     0,     0,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   307,     0,   288,   289,     0,     0,
       0,     0,     0,     0,   259,     0,     0,     0,     0,     0,
       0,   561,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   261,   262,   263,     0,     0,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,     0,     0,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   562,     0,   288,   289,     0,     0,
       0,     0,     0,     0,   259,     0,     0,     0,     0,     0,
       0,   789,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   261,   262,   263,     0,     0,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,     0,     0,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   790,     0,   288,   289,     0,     0,
       0,   314,     0,     0,   259,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   261,   262,   263,     0,     0,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,     0,     0,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   259,   317,   288,   289,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   261,   262,   263,     0,     0,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,     0,   325,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,     0,     0,   288,   289,   259,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   261,   262,   263,
     331,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   259,     0,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,     0,     0,
     288,   289,     0,     0,   261,   262,   263,   350,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,     0,   259,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,     0,     0,   288,   289,     0,
       0,     0,   261,   262,   263,   361,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     259,     0,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,     0,     0,   288,   289,     0,     0,   261,
     262,   263,   534,     0,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   351,   275,   276,     0,   259,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   286,   287,
       0,     0,   288,   289,     0,     0,     0,   261,   262,   263,
     535,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   259,     0,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,     0,     0,
     288,   289,     0,     0,   261,   262,   263,   536,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,     0,   259,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,     0,     0,   288,   289,     0,
       0,     0,   261,   262,   263,   537,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     259,     0,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,     0,     0,   288,   289,     0,     0,   261,
     262,   263,   538,     0,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,     0,   259,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   286,   287,
       0,     0,   288,   289,     0,     0,     0,   261,   262,   263,
     539,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   259,     0,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,     0,     0,
     288,   289,     0,     0,   261,   262,   263,   540,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,     0,   259,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,     0,     0,   288,   289,     0,
       0,     0,   261,   262,   263,   541,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     259,     0,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,     0,     0,   288,   289,     0,     0,   261,
     262,   263,   542,     0,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,     0,   259,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   286,   287,
       0,     0,   288,   289,     0,     0,     0,   261,   262,   263,
     543,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   259,     0,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,     0,     0,
     288,   289,     0,     0,   261,   262,   263,   548,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,     0,   259,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,     0,     0,   288,   289,     0,
       0,     0,   261,   262,   263,   553,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     259,     0,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,     0,     0,   288,   289,     0,     0,   261,
     262,   263,   563,     0,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,     0,   259,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   286,   287,
       0,     0,   288,   289,     0,     0,     0,   261,   262,   263,
     669,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   259,     0,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,     0,     0,
     288,   289,     0,     0,   261,   262,   263,   698,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,     0,   259,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,     0,     0,   288,   289,     0,
       0,     0,   261,   262,   263,   801,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     259,     0,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,     0,     0,   288,   289,     0,     0,   261,
     262,   263,   811,     0,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,     0,   259,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   286,   287,
       0,     0,   288,   289,     0,     0,     0,   261,   262,   263,
       0,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   259,     0,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,     0,     0,
     288,   289,     0,     0,   261,   262,   263,     0,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,     0,     0,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,     0,     0,   288,   289,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,     0,     0,
       0,   198,     0,     0,     0,    40,     0,    42,    43,     0,
       0,    45,   199,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   200,     0,     0,     0,
       0,     0,     0,    47,    48,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,    21,    22,     0,
       0,    54,     0,    55,    56,    57,     0,     0,    30,     6,
       7,     8,     9,    10,     0,     0,     0,   198,     0,     0,
       0,    40,     0,    42,    43,     0,     0,    45,     0,    46,
       0,    21,    22,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,     0,    47,
      48,   198,     0,     0,     0,    40,     0,    42,    43,     0,
       0,    45,     0,    46,    50,    51,     0,     0,     0,    52,
       0,     0,     0,   403,     0,     0,     0,    54,     0,    55,
      56,    57,     0,    47,    48,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,   454,    21,    22,
       0,    54,     0,    55,    56,    57,     0,     0,     0,    30,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    40,     0,    42,    43,     0,     0,    45,   193,
      46,     0,    21,    22,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    30,     0,     0,     0,     0,     0,     0,
      47,    48,     0,     0,     0,     0,    40,     0,    42,    43,
       0,     0,    45,   366,    46,    50,    51,     0,     0,     0,
      52,     0,     0,     0,     0,     0,     0,     0,    54,     0,
      55,    56,    57,     0,    47,    48,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,    50,
      51,     0,     0,     0,    52,     0,     0,     0,    21,    22,
       0,     0,    54,     0,    55,    56,    57,     0,     0,    30,
       6,     7,     8,     9,    10,     0,     0,     0,     0,   418,
       0,     0,    40,     0,    42,    43,     0,     0,    45,     0,
      46,     0,    21,    22,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    30,     0,     0,     0,     0,     0,     0,
      47,    48,     0,     0,     0,     0,    40,     0,    42,    43,
       0,   423,    45,     0,    46,    50,    51,     0,     0,     0,
      52,     0,     6,     7,     8,     9,    10,     0,    54,     0,
      55,    56,   419,     0,    47,    48,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,    50,
      51,     0,     0,     0,    52,    30,     6,     7,     8,     9,
      10,     0,    54,     0,    55,    56,    57,     0,    40,     0,
      42,    43,     0,     0,    45,   488,    46,     0,    21,    22,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    30,
       0,     0,     0,     0,     0,     0,    47,    48,     0,     0,
       0,     0,    40,     0,    42,    43,     0,     0,    45,     0,
      46,    50,    51,     0,     0,     0,    52,     0,     6,     7,
       8,     9,    10,     0,    54,     0,    55,    56,    57,     0,
      47,    48,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,     0,     0,     0,    50,    51,     0,     0,     0,
      52,    30,     0,     0,     0,     0,     0,     0,    54,     0,
      55,    56,    57,     0,    40,     0,    42,    43,     0,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,    48,     0,     0,   369,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   259,    50,    51,     0,
       0,     0,    52,     0,     0,     0,     0,     0,     0,     0,
      54,   370,    55,    56,   547,   261,   262,   263,     0,     0,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,     0,     0,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   286,   287,   369,     0,   288,   289,
       0,     0,     0,     0,     0,     0,   259,   533,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   261,   262,   263,     0,     0,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,     0,     0,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   286,   287,   369,     0,   288,   289,
       0,     0,     0,     0,     0,     0,   259,   557,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   261,   262,   263,     0,     0,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   365,   259,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   286,   287,     0,     0,   288,   289,
       0,     0,     0,   261,   262,   263,     0,     0,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   259,   487,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,     0,     0,   288,   289,     0,     0,
     261,   262,   263,     0,     0,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   259,     0,
     277,   278,   279,   280,   281,   282,   283,   284,   285,   286,
     287,     0,     0,   288,   289,   559,     0,   261,   262,   263,
       0,     0,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   259,   582,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,     0,     0,
     288,   289,     0,     0,   261,   262,   263,     0,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,     0,     0,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,   259,     0,   288,   289,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     583,     0,     0,     0,   261,   262,   263,     0,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   259,   627,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,     0,     0,   288,   289,     0,
       0,   261,   262,   263,     0,     0,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   259,
     638,   277,   278,   279,   280,   281,   282,   283,   284,   285,
     286,   287,     0,     0,   288,   289,     0,     0,   261,   262,
     263,     0,     0,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   259,     0,   277,   278,
     279,   280,   281,   282,   283,   284,   285,   286,   287,     0,
       0,   288,   289,     0,     0,   261,   262,   263,     0,     0,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   259,     0,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   286,   287,     0,     0,   288,   289,
       0,     0,     0,   262,   263,     0,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     259,     0,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,     0,     0,   288,   289,     0,     0,     0,
       0,   263,     0,     0,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   259,     0,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   286,   287,
       0,     0,   288,   289,     0,     0,     0,     0,     0,     0,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,     0,     0,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,     0,     0,   288,
     289
};

static const yytype_int16 yycheck[] =
{
       2,   346,   237,   633,     2,   111,    12,    56,     3,     1,
     224,    17,   226,    55,    46,    17,   230,    49,   232,     1,
       6,    53,   119,     9,     6,   122,     1,     6,     3,     4,
      13,     6,    74,    16,     3,    18,   133,    20,     3,     1,
      23,     3,    25,     1,     6,    28,    48,   144,     6,    32,
       1,     3,    35,     1,     3,     6,    39,    40,     6,   108,
     109,     0,    45,    46,    47,    57,    49,    50,    51,    52,
      53,    54,    55,    56,    57,     3,   173,    79,     6,     1,
      82,    79,    74,     3,    82,    87,     6,     1,     1,    87,
       3,     4,     5,     6,     7,     8,    78,    55,     3,    78,
     102,    46,     1,     1,   102,    74,     1,     1,   205,     1,
      38,    48,    74,    26,    27,    90,    74,     1,    38,    47,
      48,    90,    74,    71,    37,    74,    74,    47,    48,     1,
       3,     3,     4,     3,     6,   241,     3,    50,   768,    52,
      53,     3,     3,    56,     3,    58,     3,     3,     3,     3,
       3,    73,    74,    87,   156,     3,    55,     3,   156,    57,
      55,     1,    56,    55,    92,    78,    79,     3,     3,    74,
      74,    55,    92,    46,    46,    74,    74,     3,   392,    74,
      93,    94,    74,    87,     1,    98,     3,     3,     3,    46,
      74,   566,     6,   106,    56,   108,   109,   110,   573,    58,
      59,    74,    74,     1,    74,     3,    56,    74,     3,     3,
      46,    46,   195,    74,     3,   198,    56,    74,    74,    74,
      74,    74,     3,     1,   256,     3,    74,   259,    74,    46,
      46,    46,   467,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,     1,    74,   108,   109,    46,     6,
       7,    46,    46,   255,   104,   105,     6,    46,   108,   109,
     243,    90,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,     1,   256,     3,     3,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   295,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,   311,   520,   290,     1,    56,
       3,     3,   657,   296,     1,     1,     3,    46,     1,   416,
       6,     4,     5,     3,     7,     8,     1,   310,     3,   312,
       1,     3,     3,     1,     3,     3,     3,   320,   321,     1,
      33,     1,     3,     3,     6,     3,     6,     1,     3,     3,
       4,     5,     6,     7,     8,     1,   103,   104,   105,    46,
       6,   108,   109,    56,    26,    27,    56,    87,   351,    52,
      53,    46,    26,    27,    46,    46,   590,   591,    46,    46,
     382,   383,   384,    37,     3,   387,   369,   370,     4,   391,
       6,   393,   375,   376,   377,   393,    50,   632,    52,    53,
       3,     1,    56,     3,    58,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    56,     6,   108,   109,
     403,     3,    58,    59,    78,    79,    74,     1,     1,     3,
       3,   776,    56,    33,     3,     9,   419,    26,    27,    93,
      94,    14,    90,     1,    98,     1,     3,     3,     6,    23,
      24,    87,   106,     3,   108,   109,   110,    56,    98,    99,
     100,   101,   102,   103,   104,   105,   680,     3,   108,   109,
       3,   454,    46,    97,    98,    99,   100,   101,   102,   103,
     104,   105,     1,     3,   108,   109,     3,     6,   702,    88,
      89,    90,    91,    92,     3,   478,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,     3,    56,   108,
     109,    58,    59,   545,     1,   498,     1,    58,    59,     6,
     734,     6,   524,   737,     4,     4,     6,     6,   742,     1,
     744,   628,     4,    86,     6,     7,     8,    58,    59,    86,
      87,     3,   544,     3,     1,    57,     3,     4,     5,     6,
       7,     8,   100,   101,   102,   103,   104,   105,     3,     3,
     108,   109,   545,     1,   547,     3,     1,     1,     1,    26,
      27,     9,   555,     3,     3,   558,   559,     3,   792,     6,
      37,   795,     6,     6,   798,    23,    24,     4,   802,     6,
       7,     8,     6,    50,     3,    52,    53,    74,    71,    56,
     583,    58,     6,     6,    57,     6,     3,     3,    46,   615,
       4,     5,     6,     7,     8,     3,    87,     3,   620,     6,
      33,    78,    79,     3,     6,     3,     3,     3,     3,     3,
       9,     3,     3,    27,     3,    30,    93,    94,     3,   641,
     642,    98,     9,   641,   642,     9,     9,    55,     3,   106,
      56,   108,   109,   110,    74,     3,     3,     3,    52,    53,
       3,    73,     3,     3,     3,     3,    55,     3,    87,   671,
     672,    55,     3,    46,     6,     9,     6,     4,     3,   681,
       9,     3,    87,   681,     3,     9,     6,   693,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
       3,    55,     3,     3,    74,     3,     3,     3,     3,     3,
     712,     3,     3,     6,   712,     3,     3,    87,   672,   712,
     386,   678,   518,   706,   687,   692,   477,   358,   469,   659,
     776,   573,   573,   735,   573,   718,   738,   735,   740,   475,
     738,   743,    32,   745,    -1,   743,    -1,   745,    -1,    -1,
      -1,   753,    -1,    -1,    -1,   757,    -1,    -1,   760,   757,
      -1,   763,   760,    -1,    -1,   763,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   778,    -1,   780,    -1,
     782,    -1,   784,    -1,    -1,    -1,   769,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   807,   808,   809,   810,     0,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    39,    40,
      41,    42,    43,    44,    45,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    19,    20,
      21,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    16,    17,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    30,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    30,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
       1,    -1,     3,     4,     5,     6,     7,     8,    -1,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    53,    54,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    37,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,   108,   109,   110,
      -1,    50,    -1,    52,    53,    -1,    -1,    56,    57,    58,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    93,    94,    -1,    -1,     1,    98,
      37,     4,     5,     6,     7,     8,    -1,   106,    -1,   108,
     109,   110,    -1,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    78,    79,    -1,    -1,    -1,    -1,    50,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    93,    94,    -1,    -1,
       1,    98,    -1,     4,     5,     6,     7,     8,    -1,   106,
      -1,   108,   109,   110,    -1,    78,    79,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      93,    94,    -1,    -1,     1,    98,    37,     4,     5,     6,
       7,     8,    -1,   106,    -1,   108,   109,   110,    -1,    50,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      37,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    -1,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    93,    94,    -1,    -1,     1,    98,    -1,     4,
       5,     6,     7,     8,    -1,   106,    -1,   108,   109,   110,
      -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    93,    94,    -1,    -1,
       1,    98,    37,     4,     5,     6,     7,     8,    -1,   106,
      -1,   108,   109,   110,    -1,    50,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    26,    27,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,
      -1,    -1,    -1,    78,    79,    -1,    -1,    -1,    -1,    50,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    93,    94,
      -1,    -1,     1,    98,    -1,     4,     5,     6,     7,     8,
      -1,   106,    -1,   108,   109,   110,    -1,    78,    79,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    93,    94,    -1,    -1,     1,    98,    37,     4,
       5,     6,     7,     8,    -1,   106,    -1,   108,   109,   110,
      -1,    50,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    78,
      79,    -1,    -1,    -1,    -1,    50,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    93,    94,    -1,    -1,     1,    98,
      -1,     4,     5,     6,     7,     8,    -1,   106,    -1,   108,
     109,   110,    -1,    78,    79,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    93,    94,
      -1,    -1,     1,    98,    37,     4,     5,     6,     7,     8,
      -1,   106,    -1,   108,   109,   110,    -1,    50,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    26,    27,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,
      -1,    -1,    -1,    -1,    -1,    78,    79,    -1,    -1,    -1,
      -1,    50,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      93,    94,    -1,    -1,     1,    98,    -1,     4,     5,     6,
       7,     8,    -1,   106,    -1,   108,   109,   110,    -1,    78,
      79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    93,    94,    -1,    -1,     1,    98,
      37,     4,     5,     6,     7,     8,    -1,   106,    -1,   108,
     109,   110,    -1,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    78,    79,    -1,    -1,    -1,    -1,    50,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    93,    94,    -1,    -1,
       1,    98,    -1,     4,     5,     6,     7,     8,    -1,   106,
      -1,   108,   109,   110,    -1,    78,    79,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      93,    94,    -1,    -1,     1,    98,    37,     4,     5,     6,
       7,     8,    -1,   106,    -1,   108,   109,   110,    -1,    50,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      37,    -1,    -1,     1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    -1,    50,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   106,    -1,   108,   109,   110,
      -1,    78,    79,    -1,    -1,    -1,    -1,    -1,    46,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    56,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   106,
      -1,   108,   109,   110,    -1,    -1,    -1,    75,    76,    77,
      -1,    -1,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
     108,   109,     3,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    50,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,
      -1,    -1,    -1,    -1,    46,    -1,    -1,    -1,    -1,    -1,
      -1,     3,    93,    94,    56,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   106,    -1,   108,   109,   110,
      72,    -1,    -1,    75,    76,    77,    -1,    -1,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    56,    -1,   108,   109,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    70,    -1,
      -1,    -1,    -1,    75,    76,    77,    -1,    -1,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    46,    -1,   108,   109,    -1,    -1,
      -1,    -1,    -1,    -1,    56,    -1,    -1,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    75,    76,    77,    -1,    -1,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    46,    -1,   108,   109,    -1,    -1,
      -1,    -1,    -1,    -1,    56,    -1,    -1,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    75,    76,    77,    -1,    -1,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    46,    -1,   108,   109,    -1,    -1,
      -1,    -1,    -1,    -1,    56,    -1,    -1,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    75,    76,    77,    -1,    -1,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    46,    -1,   108,   109,    -1,    -1,
      -1,     3,    -1,    -1,    56,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    75,    76,    77,    -1,    -1,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    56,     3,   108,   109,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    75,    76,    77,    -1,    -1,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,     3,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    -1,   108,   109,    56,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    75,    76,    77,
       3,    -1,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    56,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
     108,   109,    -1,    -1,    75,    76,    77,     3,    -1,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    56,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,   108,   109,    -1,
      -1,    -1,    75,    76,    77,     3,    -1,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      56,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,   108,   109,    -1,    -1,    75,
      76,    77,     3,    -1,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    56,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,   108,   109,    -1,    -1,    -1,    75,    76,    77,
       3,    -1,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    56,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
     108,   109,    -1,    -1,    75,    76,    77,     3,    -1,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    56,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,   108,   109,    -1,
      -1,    -1,    75,    76,    77,     3,    -1,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      56,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,   108,   109,    -1,    -1,    75,
      76,    77,     3,    -1,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    56,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,   108,   109,    -1,    -1,    -1,    75,    76,    77,
       3,    -1,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    56,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
     108,   109,    -1,    -1,    75,    76,    77,     3,    -1,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    56,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,   108,   109,    -1,
      -1,    -1,    75,    76,    77,     3,    -1,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      56,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,   108,   109,    -1,    -1,    75,
      76,    77,     3,    -1,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    56,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,   108,   109,    -1,    -1,    -1,    75,    76,    77,
       3,    -1,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    56,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
     108,   109,    -1,    -1,    75,    76,    77,     3,    -1,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    56,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,   108,   109,    -1,
      -1,    -1,    75,    76,    77,     3,    -1,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      56,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,   108,   109,    -1,    -1,    75,
      76,    77,     3,    -1,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    56,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,   108,   109,    -1,    -1,    -1,    75,    76,    77,
       3,    -1,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    56,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
     108,   109,    -1,    -1,    75,    76,    77,     3,    -1,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    56,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,   108,   109,    -1,
      -1,    -1,    75,    76,    77,     3,    -1,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      56,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,   108,   109,    -1,    -1,    75,
      76,    77,     3,    -1,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    56,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,   108,   109,    -1,    -1,    -1,    75,    76,    77,
      -1,    -1,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    56,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
     108,   109,    -1,    -1,    75,    76,    77,    -1,    -1,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,   108,   109,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    46,    -1,    -1,    -1,    50,    -1,    52,    53,    -1,
      -1,    56,    57,    58,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    -1,    -1,    -1,
      -1,    -1,    -1,    78,    79,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    26,    27,    -1,
      -1,   106,    -1,   108,   109,   110,    -1,    -1,    37,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    46,    -1,    -1,
      -1,    50,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    78,
      79,    46,    -1,    -1,    -1,    50,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,   102,    -1,    -1,    -1,   106,    -1,   108,
     109,   110,    -1,    78,    79,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,   102,    26,    27,
      -1,   106,    -1,   108,   109,   110,    -1,    -1,    -1,    37,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    50,    -1,    52,    53,    -1,    -1,    56,    57,
      58,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,
      78,    79,    -1,    -1,    -1,    -1,    50,    -1,    52,    53,
      -1,    -1,    56,    57,    58,    93,    94,    -1,    -1,    -1,
      98,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   106,    -1,
     108,   109,   110,    -1,    78,    79,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    26,    27,
      -1,    -1,   106,    -1,   108,   109,   110,    -1,    -1,    37,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    47,
      -1,    -1,    50,    -1,    52,    53,    -1,    -1,    56,    -1,
      58,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,
      78,    79,    -1,    -1,    -1,    -1,    50,    -1,    52,    53,
      -1,    55,    56,    -1,    58,    93,    94,    -1,    -1,    -1,
      98,    -1,     4,     5,     6,     7,     8,    -1,   106,    -1,
     108,   109,   110,    -1,    78,    79,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    37,     4,     5,     6,     7,
       8,    -1,   106,    -1,   108,   109,   110,    -1,    50,    -1,
      52,    53,    -1,    -1,    56,    57,    58,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,
      -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,    -1,
      -1,    -1,    50,    -1,    52,    53,    -1,    -1,    56,    -1,
      58,    93,    94,    -1,    -1,    -1,    98,    -1,     4,     5,
       6,     7,     8,    -1,   106,    -1,   108,   109,   110,    -1,
      78,    79,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    37,    -1,    -1,    -1,    -1,    -1,    -1,   106,    -1,
     108,   109,   110,    -1,    50,    -1,    52,    53,    -1,    -1,
      56,    -1,    58,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    78,    79,    -1,    -1,    46,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    56,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     106,    71,   108,   109,   110,    75,    76,    77,    -1,    -1,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    46,    -1,   108,   109,
      -1,    -1,    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    75,    76,    77,    -1,    -1,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    46,    -1,   108,   109,
      -1,    -1,    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    75,    76,    77,    -1,    -1,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    55,    56,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,    -1,   108,   109,
      -1,    -1,    -1,    75,    76,    77,    -1,    -1,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    56,    57,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    -1,   108,   109,    -1,    -1,
      75,    76,    77,    -1,    -1,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    56,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    -1,   108,   109,    73,    -1,    75,    76,    77,
      -1,    -1,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    56,    57,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
     108,   109,    -1,    -1,    75,    76,    77,    -1,    -1,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    56,    -1,   108,   109,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      71,    -1,    -1,    -1,    75,    76,    77,    -1,    -1,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    56,    57,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,   108,   109,    -1,
      -1,    75,    76,    77,    -1,    -1,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    56,
      57,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,   108,   109,    -1,    -1,    75,    76,
      77,    -1,    -1,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    56,    -1,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,    -1,
      -1,   108,   109,    -1,    -1,    75,    76,    77,    -1,    -1,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    56,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,    -1,   108,   109,
      -1,    -1,    -1,    76,    77,    -1,    -1,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      56,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,   108,   109,    -1,    -1,    -1,
      -1,    77,    -1,    -1,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    56,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,   108,   109,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    -1,    -1,   108,
     109
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   112,   113,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    45,    47,    49,
      50,    51,    52,    53,    54,    56,    58,    78,    79,    83,
      93,    94,    98,   104,   106,   108,   109,   110,   114,   115,
     116,   117,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   136,   137,
     138,   140,   141,   149,   150,   151,   153,   154,   155,   160,
     161,   162,   169,   171,   184,   186,   195,   196,   198,   205,
     206,   207,   209,   211,   219,   220,   221,   222,   224,   227,
     228,   229,   232,   250,   255,   259,   260,   261,   262,   263,
     264,   265,   266,   271,   274,   275,   276,     3,     3,     1,
     118,   261,     1,   263,   264,     1,     3,     1,     3,    14,
       1,   264,     1,   261,   263,   279,     1,   264,     1,     1,
     264,     1,   264,   277,     1,     3,    46,     1,   264,     1,
       6,     1,     6,     1,     3,   264,   256,   272,     1,     6,
       7,     1,   264,   266,     1,     6,     1,     3,    46,     1,
     264,     1,     3,     6,   223,     1,     6,   225,   226,     1,
       6,     1,   264,    57,   264,   278,    46,   264,    46,    57,
      71,   264,   277,   281,   264,   263,     1,     3,   277,   264,
     264,   264,     1,     3,   277,   264,   264,   264,   264,   135,
      32,    34,    47,   117,   139,   117,   152,   117,   170,   185,
     197,    48,   214,   217,   218,   117,     1,    56,     1,     6,
     230,   231,   230,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    74,    87,   265,     3,    56,
      70,    75,    76,    77,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   108,   109,
      58,    59,   265,     3,     3,    74,    87,     3,    46,     3,
      46,     3,     3,     3,     3,    46,     3,    46,     3,    46,
      87,    74,    90,     3,     3,     3,     3,     3,     3,     1,
      73,    74,     3,   117,     3,     3,     3,   233,     3,   251,
       3,     3,     6,   257,   258,     1,     6,   212,   213,   273,
       3,     3,     3,     3,     3,     3,    87,     3,    46,     3,
       3,    90,     3,     3,    74,     3,    86,     3,    74,     3,
       3,     3,     1,    57,   264,    55,    57,   264,    57,    46,
      71,     1,    57,     1,    57,    74,    86,    87,     3,     3,
       3,     3,   148,   148,   148,   172,   187,   148,     1,     3,
      46,   148,   215,   216,     3,     1,   212,     3,     3,    74,
       9,   230,     3,   102,   264,     6,   264,   264,   264,   264,
     264,   264,   264,   264,   264,   264,   263,   280,    47,   110,
     264,   268,   277,    55,   277,   264,     1,   264,   264,   264,
     264,   264,   264,   264,   264,   264,   264,   264,     6,   264,
     264,   264,   264,   264,   264,   264,   264,   264,   264,   264,
     264,   264,   264,   264,   102,   264,     6,   261,   264,   264,
     261,     1,   264,     3,   264,   264,     1,    56,   234,   235,
       1,    33,   237,   252,     3,    74,     3,    74,    71,     1,
     260,     1,   264,     6,     4,     6,   226,    57,    57,   264,
     264,    57,   264,   264,   264,     9,   117,    16,    17,   142,
     144,   145,   147,     9,     1,     3,    23,    24,   173,   178,
     180,     1,     3,    23,   178,   188,    30,   199,   200,   201,
     202,     3,    46,     9,   148,   117,   210,     1,    55,     6,
       3,     3,   264,    57,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,    74,    87,   269,   110,     3,     3,
       3,     1,    55,     3,     1,    46,   264,    57,    87,    73,
       3,     3,    46,     3,     3,   212,   243,   237,     3,     6,
     238,   239,     3,   253,     1,   258,   213,   264,     3,     3,
       3,     3,    57,    71,     3,     1,     3,     1,   264,     9,
     143,   146,     3,     3,     1,     4,     6,     7,     8,   182,
     183,     1,     9,   179,     3,     1,     4,     6,   193,   194,
       9,     1,     3,     4,     6,    90,   203,   204,     9,   201,
     148,     3,     9,    55,   208,     3,    46,    57,   263,   264,
     277,     1,    56,   270,     3,   267,     1,   264,    57,   264,
     264,   156,   157,     1,    55,     3,     6,    38,    47,    48,
      92,   206,   244,   245,   247,   248,     3,    56,   240,    74,
       3,   206,   245,   247,   248,   254,   264,     3,     3,     3,
       3,   148,   148,     3,    46,    73,     3,    46,    74,     3,
       3,    46,   181,     3,    46,     3,    46,    74,     3,     3,
     261,     3,    74,    90,     3,     3,    46,    55,     3,     3,
       3,   212,   214,    55,     3,    46,    72,    19,    20,    21,
     117,   158,   159,   163,   165,   167,   117,   236,    87,     3,
       6,     1,     6,    78,   249,     9,     6,    27,   241,   242,
     260,   239,     9,   142,   176,   177,     4,   174,   175,   183,
     148,   117,   191,   192,   189,   190,   194,     3,   204,   261,
       3,     1,    55,   148,   264,     1,     3,    46,     1,     3,
      46,     1,     3,    46,     9,   158,    55,   264,   246,    87,
       3,     6,     3,    74,     3,    55,    74,     3,   148,   117,
     148,   117,   148,   117,   148,   117,     3,     3,     9,     3,
      46,     3,   164,   117,     3,   166,   117,     3,   168,   117,
       3,     3,   214,   264,     6,    78,   242,   148,   148,   148,
     148,     3,     6,     9,     9,     9,     9,     3,     3,     3,
       3
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
        case 6:
#line 206 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); ;}
    break;

  case 7:
#line 207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); ;}
    break;

  case 8:
#line 213 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      ;}
    break;

  case 10:
#line 219 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 11:
#line 224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 12:
#line 229 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 13:
#line 234 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 17:
#line 245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 18:
#line 251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 19:
#line 257 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      ;}
    break;

  case 20:
#line 264 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); ;}
    break;

  case 21:
#line 265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; ;}
    break;

  case 22:
#line 268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; ;}
    break;

  case 23:
#line 269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; ;}
    break;

  case 24:
#line 270 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; ;}
    break;

  case 25:
#line 271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;;}
    break;

  case 26:
#line 276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   ;}
    break;

  case 27:
#line 281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 28:
#line 288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); ;}
    break;

  case 50:
#line 314 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; ;}
    break;

  case 51:
#line 316 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); ;}
    break;

  case 52:
#line 321 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 53:
#line 325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtUnref( LINE, (yyvsp[(1) - (5)].fal_val) );
   ;}
    break;

  case 54:
#line 329 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) );
      ;}
    break;

  case 55:
#line 333 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 56:
#line 337 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), (yyvsp[(5) - (6)].fal_val) );
      ;}
    break;

  case 57:
#line 342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) ) );
      ;}
    break;

  case 69:
#line 366 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoAdd( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 70:
#line 373 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSub( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 71:
#line 380 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMul( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 72:
#line 387 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoDiv( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 73:
#line 394 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMod( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 74:
#line 401 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoPow( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 75:
#line 408 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBAND( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 76:
#line 415 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 77:
#line 422 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBXOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 78:
#line 428 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHL( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 79:
#line 434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 80:
#line 442 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      ;}
    break;

  case 81:
#line 449 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 82:
#line 456 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      ;}
    break;

  case 83:
#line 464 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 84:
#line 465 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 85:
#line 466 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; ;}
    break;

  case 86:
#line 470 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 87:
#line 471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 88:
#line 472 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 89:
#line 476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      ;}
    break;

  case 90:
#line 484 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 91:
#line 491 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 92:
#line 501 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 93:
#line 502 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; ;}
    break;

  case 94:
#line 506 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 95:
#line 507 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 98:
#line 514 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      ;}
    break;

  case 101:
#line 524 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); ;}
    break;

  case 102:
#line 529 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      ;}
    break;

  case 104:
#line 541 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 105:
#line 542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; ;}
    break;

  case 107:
#line 547 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   ;}
    break;

  case 108:
#line 554 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      ;}
    break;

  case 109:
#line 563 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 110:
#line 571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      ;}
    break;

  case 111:
#line 581 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      ;}
    break;

  case 112:
#line 590 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 113:
#line 597 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>( (yyvsp[(1) - (1)].fal_stat) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 114:
#line 604 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 115:
#line 612 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (2)].fal_stat) != 0 )
         {
            Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>((yyvsp[(1) - (2)].fal_stat));
            if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
                f->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
            (yyval.fal_stat) = f;
         }
         else
            delete (yyvsp[(2) - (2)].fal_stat);
      ;}
    break;

  case 116:
#line 627 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 117:
#line 631 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 118:
#line 636 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, 0, 0, 0 );
      ;}
    break;

  case 119:
#line 643 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 120:
#line 647 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 121:
#line 652 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for, "", CURRENT_LINE );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, 0, 0, 0 );
      ;}
    break;

  case 122:
#line 661 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (5)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (5)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( LINE, new Falcon::Value(decl), (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 123:
#line 677 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 124:
#line 685 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
          Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (5)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( CURRENT_LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (5)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( CURRENT_LINE, new Falcon::Value(decl), (yyvsp[(4) - (5)].fal_val) );

         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 125:
#line 701 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(7) - (7)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(7) - (7)].fal_stat) );
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 126:
#line 711 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_forin ); ;}
    break;

  case 129:
#line 720 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 133:
#line 734 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getLoop());
         if ( f == 0 || f->type() != Falcon::Statement::t_forin )
         {
            COMPILER->raiseError( Falcon::e_syn_fordot );
            delete (yyvsp[(2) - (3)].fal_val);
            (yyval.fal_stat) = 0;
         }
         else {
            (yyval.fal_stat) = new Falcon::StmtFordot( LINE, (yyvsp[(2) - (3)].fal_val) );
         }
      ;}
    break;

  case 134:
#line 747 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 135:
#line 755 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 136:
#line 759 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 137:
#line 765 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 138:
#line 771 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      ;}
    break;

  case 139:
#line 778 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 140:
#line 783 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 141:
#line 792 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   ;}
    break;

  case 142:
#line 801 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      ;}
    break;

  case 143:
#line 813 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 144:
#line 815 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 145:
#line 824 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 146:
#line 828 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      ;}
    break;

  case 147:
#line 840 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 148:
#line 841 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 149:
#line 850 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 150:
#line 854 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 // Apparently you get a segfault without it.
		 // (Note that the formiddle: version below does *not* need it
		 f->middleBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->middleBlock() );
      ;}
    break;

  case 151:
#line 868 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 152:
#line 870 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->middleBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 153:
#line 879 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); ;}
    break;

  case 154:
#line 883 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 155:
#line 891 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 156:
#line 900 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 157:
#line 902 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 160:
#line 911 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 162:
#line 917 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 164:
#line 927 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 165:
#line 935 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 166:
#line 939 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 168:
#line 951 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 169:
#line 961 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 171:
#line 970 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      ;}
    break;

  case 175:
#line 984 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 177:
#line 988 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 180:
#line 1000 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 181:
#line 1009 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 182:
#line 1021 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].stringp) );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 183:
#line 1032 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (yyvsp[(1) - (3)].integer) ), new Falcon::Value( (yyvsp[(3) - (3)].integer) ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 184:
#line 1043 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 185:
#line 1063 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 186:
#line 1071 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 187:
#line 1080 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 188:
#line 1082 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 191:
#line 1091 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 193:
#line 1097 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 195:
#line 1107 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 196:
#line 1116 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 197:
#line 1120 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 199:
#line 1132 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 200:
#line 1142 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 204:
#line 1156 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 205:
#line 1168 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 206:
#line 1189 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_val), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 207:
#line 1193 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 208:
#line 1197 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 209:
#line 1205 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 210:
#line 1212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 211:
#line 1222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 213:
#line 1231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 219:
#line 1251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }
         // but continue by pushing this new context
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      ;}
    break;

  case 220:
#line 1269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }

         // but continue by pushing this new context
         COMPILER->defineVal( (yyvsp[(3) - (4)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(3) - (4)].fal_val) );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      ;}
    break;

  case 221:
#line 1289 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 222:
#line 1299 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 223:
#line 1310 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 226:
#line 1323 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 227:
#line 1335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 228:
#line 1357 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 229:
#line 1358 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 230:
#line 1370 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 231:
#line 1376 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 233:
#line 1385 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 234:
#line 1386 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 235:
#line 1389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 237:
#line 1394 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 238:
#line 1395 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 239:
#line 1402 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String *func_name;
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::String complete_name = stmt_cls->symbol()->name() + "." + *(yyvsp[(2) - (2)].stringp);
            func_name = COMPILER->addString( complete_name );
         }
         else
            func_name = (yyvsp[(2) - (2)].stringp);

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( func_name );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( func_name );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         // and eventually add it as a class property
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
            if ( cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) ) {
               COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (2)].stringp) );
            }
            else {
                cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef( sym ) );
            }
         }

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 243:
#line 1463 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), (yyvsp[(1) - (1)].stringp) );
            COMPILER->module()->addSymbol( sym );
            func->addParameter( sym );
         }
      ;}
    break;

  case 245:
#line 1480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 246:
#line 1486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 247:
#line 1491 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 248:
#line 1497 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 250:
#line 1506 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 252:
#line 1511 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 253:
#line 1521 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 254:
#line 1524 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 255:
#line 1533 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 256:
#line 1540 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );

         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
      ;}
    break;

  case 257:
#line 1550 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 258:
#line 1556 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 259:
#line 1568 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = (yyvsp[(4) - (5)].fal_val); //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *(yyvsp[(2) - (5)].stringp), val, LINE );
         // we don't need the expression anymore
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 260:
#line 1578 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 261:
#line 1583 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 262:
#line 1595 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 263:
#line 1604 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 264:
#line 1611 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 265:
#line 1619 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 266:
#line 1624 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 267:
#line 1636 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 268:
#line 1641 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     ;}
    break;

  case 271:
#line 1654 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      ;}
    break;

  case 272:
#line 1658 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      ;}
    break;

  case 273:
#line 1672 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 274:
#line 1679 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 276:
#line 1687 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 278:
#line 1691 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 280:
#line 1697 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 281:
#line 1701 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 284:
#line 1710 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 285:
#line 1721 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );
      ;}
    break;

  case 286:
#line 1755 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      ;}
    break;

  case 288:
#line 1783 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 291:
#line 1791 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 292:
#line 1792 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 297:
#line 1809 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol((yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            if ( (yyvsp[(2) - (2)].fal_adecl) != 0 )
            {
               // save the carried
               Falcon::ListElement *iter = (yyvsp[(2) - (2)].fal_adecl)->begin();
               while( iter != 0 )
               {
                  Falcon::Value *val = (Falcon::Value *) iter->data();
                  idef->addParameter( val->genVarDefSym() );
                  iter = iter->next();
               }

               // dispose of the carrier
               delete (yyvsp[(2) - (2)].fal_adecl);
            }
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
         }
      ;}
    break;

  case 298:
#line 1842 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 299:
#line 1847 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
   ;}
    break;

  case 300:
#line 1853 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 301:
#line 1854 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 303:
#line 1860 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      ;}
    break;

  case 304:
#line 1868 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 308:
#line 1878 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 309:
#line 1881 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   ;}
    break;

  case 311:
#line 1903 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         if( cls->initGiven() ) {
            COMPILER->raiseError(Falcon::e_init_given );
         }
         else
         {
            cls->initGiven( true );
            Falcon::StmtFunction *func = cls->ctorFunction();
            if ( func == 0 ) {
               func = COMPILER->buildCtorFor( cls );
            }

            // prepare the statement allocation context
            COMPILER->pushContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      ;}
    break;

  case 312:
#line 1927 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 313:
#line 1938 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(4) - (5)].fal_val)->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *(yyvsp[(2) - (5)].stringp);
         Falcon::Symbol *sym = COMPILER->addGlobalVar( COMPILER->addString(prop_name), def );
         if( clsdef->hasProperty( *(yyvsp[(2) - (5)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (5)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(2) - (5)].stringp), new Falcon::VarDef( Falcon::VarDef::t_reference, sym) );
      }
      else {
         COMPILER->raiseError(Falcon::e_static_const );
      }
      delete (yyvsp[(4) - (5)].fal_val); // the expression is not needed anymore
      (yyval.fal_stat) = 0; // we don't add any statement
   ;}
    break;

  case 314:
#line 1960 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(3) - (4)].fal_val)->genVarDef();

      if ( def != 0 ) {
         if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), def );
         delete (yyvsp[(3) - (4)].fal_val); // the expression is not needed anymore
         (yyval.fal_stat) = 0; // we don't add any statement
      }
      else {
         // create anyhow a nil property
          if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), new Falcon::VarDef() );
         // but also prepare a statement to be executed by the auto-constructor.
         (yyval.fal_stat) = new Falcon::StmtVarDef( LINE, (yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
   ;}
    break;

  case 317:
#line 1990 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 318:
#line 1997 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 319:
#line 2005 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 320:
#line 2011 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 321:
#line 2017 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 322:
#line 2030 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *(yyvsp[(2) - (2)].stringp);
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( COMPILER->addString( cl_name ) );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 323:
#line 2070 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 325:
#line 2095 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 329:
#line 2107 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 330:
#line 2110 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   ;}
    break;

  case 332:
#line 2138 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 333:
#line 2143 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
      ;}
    break;

  case 336:
#line 2158 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 337:
#line 2165 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 338:
#line 2180 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 339:
#line 2181 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 340:
#line 2182 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 341:
#line 2192 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 342:
#line 2193 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); ;}
    break;

  case 343:
#line 2194 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); ;}
    break;

  case 344:
#line 2195 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 345:
#line 2196 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 346:
#line 2197 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 347:
#line 2202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            val = new Falcon::Value();
            val->setSymdef( (yyvsp[(1) - (1)].stringp) );
            // warning: the symbol is still undefined.
            COMPILER->addSymdef( val );
         }
         else {
            val = new Falcon::Value( sym );
         }
         (yyval.fal_val) = val;
     ;}
    break;

  case 349:
#line 2220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 350:
#line 2221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 352:
#line 2233 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 353:
#line 2238 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 354:
#line 2243 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 355:
#line 2249 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 358:
#line 2264 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 359:
#line 2265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 360:
#line 2266 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 361:
#line 2267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 362:
#line 2268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 363:
#line 2269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 364:
#line 2270 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 365:
#line 2271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 366:
#line 2272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 367:
#line 2273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 368:
#line 2274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 369:
#line 2275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 370:
#line 2276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 371:
#line 2277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 372:
#line 2279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 373:
#line 2281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 374:
#line 2282 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 375:
#line 2283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 376:
#line 2284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 377:
#line 2285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 378:
#line 2286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 379:
#line 2287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 380:
#line 2288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 381:
#line 2289 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 382:
#line 2290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 383:
#line 2291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 384:
#line 2292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 385:
#line 2293 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 386:
#line 2294 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 387:
#line 2295 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 388:
#line 2296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 389:
#line 2297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 390:
#line 2298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 391:
#line 2299 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 392:
#line 2300 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 393:
#line 2301 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 394:
#line 2302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 395:
#line 2303 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 398:
#line 2306 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 399:
#line 2314 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 400:
#line 2318 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 401:
#line 2322 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 406:
#line 2330 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 407:
#line 2335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 408:
#line 2338 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 409:
#line 2341 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 410:
#line 2344 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 411:
#line 2351 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      ;}
    break;

  case 412:
#line 2357 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      ;}
    break;

  case 413:
#line 2361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 414:
#line 2362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 415:
#line 2371 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 416:
#line 2404 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         ;}
    break;

  case 418:
#line 2415 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      ;}
    break;

  case 419:
#line 2419 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 420:
#line 2427 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 421:
#line 2458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->checkLocalUndefined();
            COMPILER->closeFunction();
         ;}
    break;

  case 423:
#line 2472 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      ;}
    break;

  case 424:
#line 2481 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 425:
#line 2486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 426:
#line 2493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 427:
#line 2500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 428:
#line 2509 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 429:
#line 2511 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 430:
#line 2515 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 431:
#line 2521 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 432:
#line 2523 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 433:
#line 2527 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 434:
#line 2535 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 435:
#line 2536 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); ;}
    break;

  case 436:
#line 2538 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      ;}
    break;

  case 437:
#line 2545 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 438:
#line 2546 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 439:
#line 2550 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 440:
#line 2551 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (2)].fal_adecl)->pushBack( (yyvsp[(2) - (2)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (2)].fal_adecl); ;}
    break;

  case 441:
#line 2555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 442:
#line 2561 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 443:
#line 2568 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 444:
#line 2574 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 445:
#line 2581 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 446:
#line 2582 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6345 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2586 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


