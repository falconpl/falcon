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
     FORALL = 276,
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
     OUTER_STRING = 307,
     CLOSEPAR = 308,
     OPENPAR = 309,
     CLOSESQUARE = 310,
     OPENSQUARE = 311,
     DOT = 312,
     ASSIGN_POW = 313,
     ASSIGN_SHL = 314,
     ASSIGN_SHR = 315,
     ASSIGN_BXOR = 316,
     ASSIGN_BOR = 317,
     ASSIGN_BAND = 318,
     ASSIGN_MOD = 319,
     ASSIGN_DIV = 320,
     ASSIGN_MUL = 321,
     ASSIGN_SUB = 322,
     ASSIGN_ADD = 323,
     ARROW = 324,
     FOR_STEP = 325,
     OP_TO = 326,
     COMMA = 327,
     QUESTION = 328,
     OR = 329,
     AND = 330,
     NOT = 331,
     LET = 332,
     LE = 333,
     GE = 334,
     LT = 335,
     GT = 336,
     NEQ = 337,
     EEQ = 338,
     OP_EQ = 339,
     OP_ASSIGN = 340,
     PROVIDES = 341,
     OP_NOTIN = 342,
     OP_IN = 343,
     HASNT = 344,
     HAS = 345,
     DIESIS = 346,
     ATSIGN = 347,
     CAP = 348,
     VBAR = 349,
     AMPER = 350,
     MINUS = 351,
     PLUS = 352,
     PERCENT = 353,
     SLASH = 354,
     STAR = 355,
     POW = 356,
     SHR = 357,
     SHL = 358,
     BANG = 359,
     NEG = 360,
     DECREMENT = 361,
     INCREMENT = 362,
     DOLLAR = 363
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
#define FORALL 276
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
#define OUTER_STRING 307
#define CLOSEPAR 308
#define OPENPAR 309
#define CLOSESQUARE 310
#define OPENSQUARE 311
#define DOT 312
#define ASSIGN_POW 313
#define ASSIGN_SHL 314
#define ASSIGN_SHR 315
#define ASSIGN_BXOR 316
#define ASSIGN_BOR 317
#define ASSIGN_BAND 318
#define ASSIGN_MOD 319
#define ASSIGN_DIV 320
#define ASSIGN_MUL 321
#define ASSIGN_SUB 322
#define ASSIGN_ADD 323
#define ARROW 324
#define FOR_STEP 325
#define OP_TO 326
#define COMMA 327
#define QUESTION 328
#define OR 329
#define AND 330
#define NOT 331
#define LET 332
#define LE 333
#define GE 334
#define LT 335
#define GT 336
#define NEQ 337
#define EEQ 338
#define OP_EQ 339
#define OP_ASSIGN 340
#define PROVIDES 341
#define OP_NOTIN 342
#define OP_IN 343
#define HASNT 344
#define HAS 345
#define DIESIS 346
#define ATSIGN 347
#define CAP 348
#define VBAR 349
#define AMPER 350
#define MINUS 351
#define PLUS 352
#define PERCENT 353
#define SLASH 354
#define STAR 355
#define POW 356
#define SHR 357
#define SHL 358
#define BANG 359
#define NEG 360
#define DECREMENT 361
#define INCREMENT 362
#define DOLLAR 363




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
#line 376 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 389 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   6263

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  109
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  170
/* YYNRULES -- Number of rules.  */
#define YYNRULES  442
/* YYNRULES -- Number of states.  */
#define YYNSTATES  817

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   363

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
     105,   106,   107,   108
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
    1156,  1159,  1164,  1170,  1174,  1176,  1178,  1182,  1185,  1189,
    1193,  1197,  1201,  1205,  1209,  1213,  1217,  1221,  1225,  1228,
    1233,  1238,  1242,  1245,  1248,  1251,  1254,  1258,  1262,  1266,
    1270,  1274,  1278,  1282,  1286,  1290,  1293,  1297,  1301,  1305,
    1309,  1313,  1316,  1319,  1322,  1324,  1326,  1330,  1335,  1341,
    1344,  1346,  1348,  1350,  1352,  1356,  1360,  1365,  1370,  1376,
    1381,  1385,  1386,  1393,  1394,  1401,  1406,  1410,  1413,  1414,
    1420,  1422,  1425,  1431,  1437,  1442,  1446,  1449,  1453,  1457,
    1460,  1464,  1468,  1472,  1476,  1481,  1483,  1487,  1489,  1492,
    1494,  1498,  1502
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     110,     0,    -1,   111,    -1,    -1,   111,   112,    -1,   113,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   114,    -1,
     222,    -1,   204,    -1,   230,    -1,   248,    -1,   115,    -1,
     219,    -1,   220,    -1,   225,    -1,    39,     6,     3,    -1,
      39,     7,     3,    -1,    39,     1,     3,    -1,   117,    -1,
       3,    -1,    47,     1,     3,    -1,    34,     1,     3,    -1,
      32,     1,     3,    -1,     1,     3,    -1,   259,    85,   262,
      -1,   116,    72,   259,    85,   262,    -1,   262,     3,    -1,
     118,    -1,   119,    -1,   120,    -1,   132,    -1,   149,    -1,
     153,    -1,   167,    -1,   182,    -1,   136,    -1,   147,    -1,
     148,    -1,   193,    -1,   194,    -1,   203,    -1,   257,    -1,
     253,    -1,   217,    -1,   218,    -1,   158,    -1,   159,    -1,
     160,    -1,    10,   116,     3,    -1,    10,     1,     3,    -1,
     261,    85,   262,     3,    -1,   261,    85,   108,   108,     3,
      -1,   261,    85,   275,     3,    -1,   261,    85,   266,     3,
      -1,   261,    72,   277,    85,   262,     3,    -1,   261,    72,
     277,    85,   275,     3,    -1,   121,    -1,   122,    -1,   123,
      -1,   124,    -1,   125,    -1,   126,    -1,   127,    -1,   128,
      -1,   129,    -1,   130,    -1,   131,    -1,   262,    68,   262,
       3,    -1,   261,    67,   262,     3,    -1,   261,    66,   262,
       3,    -1,   261,    65,   262,     3,    -1,   261,    64,   262,
       3,    -1,   261,    58,   262,     3,    -1,   261,    63,   262,
       3,    -1,   261,    62,   262,     3,    -1,   261,    61,   262,
       3,    -1,   261,    59,   262,     3,    -1,   261,    60,   262,
       3,    -1,    -1,   134,   133,   146,     9,     3,    -1,   135,
     115,    -1,    11,   262,     3,    -1,    51,    -1,    11,     1,
       3,    -1,    11,   262,    46,    -1,    51,    46,    -1,    11,
       1,    46,    -1,    -1,   138,   137,   146,   140,     9,     3,
      -1,   139,   115,    -1,    15,   262,     3,    -1,    15,     1,
       3,    -1,    15,   262,    46,    -1,    15,     1,    46,    -1,
      -1,   143,    -1,    -1,   142,   141,   146,    -1,    16,     3,
      -1,    16,     1,     3,    -1,    -1,   145,   144,   146,   140,
      -1,    17,   262,     3,    -1,    17,     1,     3,    -1,    -1,
     146,   115,    -1,    12,     3,    -1,    12,     1,     3,    -1,
      13,     3,    -1,    13,    14,     3,    -1,    13,     1,     3,
      -1,    -1,   151,   150,   146,     9,     3,    -1,   152,   115,
      -1,    18,   261,    85,   262,    71,   262,     3,    -1,    18,
     261,    85,   262,    71,   262,    70,   262,     3,    -1,    18,
       1,     3,    -1,    18,   261,    85,   262,    71,   262,    46,
      -1,    18,   261,    85,   262,    71,   262,    70,   262,    46,
      -1,    18,     1,    46,    -1,    -1,    18,   277,    88,   262,
       3,   154,   156,     9,     3,    -1,    -1,    18,   277,    88,
     262,    46,   155,   115,    -1,    18,   277,    88,     1,     3,
      -1,    -1,   157,   156,    -1,   115,    -1,   161,    -1,   163,
      -1,   165,    -1,    49,   262,     3,    -1,    49,     1,     3,
      -1,   102,   275,     3,    -1,   102,     3,    -1,    81,   275,
       3,    -1,    81,     3,    -1,   102,     1,     3,    -1,    81,
       1,     3,    -1,    52,    -1,    -1,    19,     3,   162,   146,
       9,     3,    -1,    19,    46,   115,    -1,    19,     1,     3,
      -1,    -1,    20,     3,   164,   146,     9,     3,    -1,    20,
      46,   115,    -1,    20,     1,     3,    -1,    -1,    21,     3,
     166,   146,     9,     3,    -1,    21,    46,   115,    -1,    21,
       1,     3,    -1,    -1,   169,   168,   170,   176,     9,     3,
      -1,    22,   262,     3,    -1,    22,     1,     3,    -1,    -1,
     170,   171,    -1,   170,     1,     3,    -1,     3,    -1,    -1,
      23,   180,     3,   172,   146,    -1,    -1,    23,   180,    46,
     173,   115,    -1,    -1,    23,     1,     3,   174,   146,    -1,
      -1,    23,     1,    46,   175,   115,    -1,    -1,    -1,   178,
     177,   179,    -1,    -1,    24,    -1,    24,     1,    -1,     3,
     146,    -1,    46,   115,    -1,   181,    -1,   180,    72,   181,
      -1,     8,    -1,     4,    -1,     7,    -1,     4,    71,     4,
      -1,     6,    -1,    -1,   184,   183,   185,   176,     9,     3,
      -1,    25,   262,     3,    -1,    25,     1,     3,    -1,    -1,
     185,   186,    -1,   185,     1,     3,    -1,     3,    -1,    -1,
      23,   191,     3,   187,   146,    -1,    -1,    23,   191,    46,
     188,   115,    -1,    -1,    23,     1,     3,   189,   146,    -1,
      -1,    23,     1,    46,   190,   115,    -1,   192,    -1,   191,
      72,   192,    -1,    -1,     4,    -1,     6,    -1,    28,   275,
      71,   262,     3,    -1,    28,   275,     1,     3,    -1,    28,
       1,     3,    -1,    29,    46,   115,    -1,    -1,   196,   195,
     146,   197,     9,     3,    -1,    29,     3,    -1,    29,     1,
       3,    -1,    -1,   198,    -1,   199,    -1,   198,   199,    -1,
     200,   146,    -1,    30,     3,    -1,    30,    88,   259,     3,
      -1,    30,   201,     3,    -1,    30,   201,    88,   259,     3,
      -1,    30,     1,     3,    -1,   202,    -1,   201,    72,   202,
      -1,     4,    -1,     6,    -1,    31,   262,     3,    -1,    31,
       1,     3,    -1,   205,   212,   146,     9,     3,    -1,   207,
     115,    -1,   209,    54,   210,    53,     3,    -1,    -1,   209,
      54,   210,     1,   206,    53,     3,    -1,   209,     1,     3,
      -1,   209,    54,   210,    53,    46,    -1,    -1,   209,    54,
       1,   208,    53,    46,    -1,    47,     6,    -1,    -1,   211,
      -1,   210,    72,   211,    -1,     6,    -1,    -1,    -1,   215,
     213,   146,     9,     3,    -1,    -1,   216,   214,   115,    -1,
      48,     3,    -1,    48,     1,     3,    -1,    48,    46,    -1,
      48,     1,    46,    -1,    40,   264,     3,    -1,    40,     1,
       3,    -1,    43,   262,     3,    -1,    43,   262,    88,   262,
       3,    -1,    43,   262,    88,     1,     3,    -1,    43,     1,
       3,    -1,    41,     6,    85,   258,     3,    -1,    41,     6,
      85,     1,     3,    -1,    41,     1,     3,    -1,    44,     3,
      -1,    44,   221,     3,    -1,    44,     1,     3,    -1,     6,
      -1,   221,    72,     6,    -1,    45,   223,     3,    -1,    45,
       1,     3,    -1,   224,    -1,   223,    72,   224,    -1,     6,
      84,     6,    -1,     6,    84,     4,    -1,   226,   229,     9,
       3,    -1,   227,   228,     3,    -1,    42,     3,    -1,    42,
       1,     3,    -1,    42,    46,    -1,    42,     1,    46,    -1,
      -1,     6,    -1,   228,    72,     6,    -1,   228,     3,    -1,
     229,   228,     3,    -1,     1,     3,    -1,    -1,    32,     6,
     231,   232,   241,   246,     9,     3,    -1,   233,   235,     3,
      -1,     1,     3,    -1,    -1,    54,   210,    53,    -1,    -1,
      54,   210,     1,   234,    53,    -1,    -1,    33,   236,    -1,
     237,    -1,   236,    72,   237,    -1,     6,   238,    -1,    -1,
      54,   239,    53,    -1,   240,    -1,   239,    72,   240,    -1,
     258,    -1,     6,    -1,    27,    -1,    -1,   241,   242,    -1,
       3,    -1,   204,    -1,   245,    -1,   243,    -1,    -1,    38,
       3,   244,   212,   146,     9,     3,    -1,    48,     6,    85,
     262,     3,    -1,     6,    85,   262,     3,    -1,    -1,    90,
     247,     3,    -1,    90,     1,     3,    -1,     6,    -1,    76,
       6,    -1,   247,    72,     6,    -1,   247,    72,    76,     6,
      -1,    -1,    34,     6,   249,   250,   251,   246,     9,     3,
      -1,   235,     3,    -1,     1,     3,    -1,    -1,   251,   252,
      -1,     3,    -1,   204,    -1,   245,    -1,   243,    -1,    -1,
      36,   254,   255,     3,    -1,   256,    -1,   255,    72,   256,
      -1,   255,    72,     1,    -1,     6,    -1,    35,     3,    -1,
      35,   262,     3,    -1,    35,     1,     3,    -1,     8,    -1,
       4,    -1,     5,    -1,     7,    -1,     6,    -1,   259,    -1,
      27,    -1,    26,    -1,   260,    -1,   261,   263,    -1,   261,
      56,   262,    55,    -1,   261,    56,   100,   262,    55,    -1,
     261,    57,     6,    -1,   258,    -1,   261,    -1,   262,    97,
     262,    -1,    96,   262,    -1,   262,    96,   262,    -1,   262,
     100,   262,    -1,   262,    99,   262,    -1,   262,    98,   262,
      -1,   262,   101,   262,    -1,   262,    95,   262,    -1,   262,
      94,   262,    -1,   262,    93,   262,    -1,   262,   103,   262,
      -1,   262,   102,   262,    -1,   104,   262,    -1,    77,   261,
      84,   262,    -1,    77,   261,    85,   262,    -1,   262,    82,
     262,    -1,   262,   107,    -1,   107,   262,    -1,   262,   106,
      -1,   106,   262,    -1,   262,    83,   262,    -1,   262,    84,
     262,    -1,   262,    85,   262,    -1,   262,    81,   262,    -1,
     262,    80,   262,    -1,   262,    79,   262,    -1,   262,    78,
     262,    -1,   262,    75,   262,    -1,   262,    74,   262,    -1,
      76,   262,    -1,   262,    90,   262,    -1,   262,    89,   262,
      -1,   262,    88,   262,    -1,   262,    87,   262,    -1,   262,
      86,     6,    -1,   108,   262,    -1,    92,   262,    -1,    91,
     262,    -1,   269,    -1,   264,    -1,   264,    57,     6,    -1,
     264,    56,   262,    55,    -1,   264,    56,   100,   262,    55,
      -1,   264,   263,    -1,   272,    -1,   273,    -1,   274,    -1,
     263,    -1,    54,   262,    53,    -1,    56,    46,    55,    -1,
      56,   262,    46,    55,    -1,    56,    46,   262,    55,    -1,
      56,   262,    46,   262,    55,    -1,   262,    54,   275,    53,
      -1,   262,    54,    53,    -1,    -1,   262,    54,   275,     1,
     265,    53,    -1,    -1,    47,   267,   268,   212,   146,     9,
      -1,    54,   210,    53,     3,    -1,    54,   210,     1,    -1,
       1,     3,    -1,    -1,    37,   270,   271,    69,   262,    -1,
     210,    -1,     1,     3,    -1,   262,    73,   262,    46,   262,
      -1,   262,    73,   262,    46,     1,    -1,   262,    73,   262,
       1,    -1,   262,    73,     1,    -1,    56,    55,    -1,    56,
     275,    55,    -1,    56,   275,     1,    -1,    50,    55,    -1,
      50,   276,    55,    -1,    50,   276,     1,    -1,    56,    69,
      55,    -1,    56,   278,    55,    -1,    56,   278,     1,    55,
      -1,   262,    -1,   275,    72,   262,    -1,   262,    -1,   276,
     262,    -1,   259,    -1,   277,    72,   259,    -1,   262,    69,
     262,    -1,   278,    72,   262,    69,   262,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   194,   194,   197,   199,   203,   204,   205,   210,   215,
     216,   221,   226,   231,   236,   237,   238,   242,   248,   254,
     262,   263,   266,   267,   268,   269,   274,   279,   286,   287,
     288,   289,   290,   291,   292,   293,   294,   295,   296,   297,
     298,   299,   300,   301,   302,   303,   304,   305,   306,   307,
     311,   313,   319,   323,   327,   331,   335,   340,   350,   351,
     352,   353,   354,   355,   356,   357,   358,   359,   360,   364,
     371,   378,   385,   392,   399,   406,   413,   420,   426,   432,
     440,   440,   454,   462,   463,   464,   468,   469,   470,   474,
     474,   489,   499,   500,   504,   505,   509,   511,   512,   512,
     521,   522,   527,   527,   539,   540,   543,   545,   551,   560,
     568,   578,   587,   595,   595,   609,   625,   629,   633,   641,
     645,   649,   659,   658,   683,   682,   708,   712,   714,   718,
     725,   726,   727,   731,   744,   752,   756,   762,   768,   775,
     780,   789,   799,   799,   813,   822,   826,   826,   839,   848,
     852,   852,   866,   875,   879,   879,   896,   897,   904,   906,
     907,   911,   913,   912,   923,   923,   935,   935,   947,   947,
     963,   966,   965,   978,   979,   980,   983,   984,   990,   991,
     995,  1004,  1016,  1027,  1038,  1059,  1059,  1076,  1077,  1084,
    1086,  1087,  1091,  1093,  1092,  1103,  1103,  1116,  1116,  1128,
    1128,  1146,  1147,  1150,  1151,  1163,  1184,  1188,  1193,  1201,
    1208,  1207,  1226,  1227,  1230,  1232,  1236,  1237,  1241,  1246,
    1264,  1284,  1294,  1305,  1313,  1314,  1318,  1330,  1353,  1354,
    1361,  1371,  1380,  1381,  1381,  1385,  1389,  1390,  1390,  1397,
    1451,  1453,  1454,  1458,  1473,  1476,  1475,  1487,  1486,  1501,
    1502,  1506,  1507,  1516,  1520,  1528,  1535,  1545,  1551,  1563,
    1573,  1578,  1590,  1599,  1606,  1614,  1619,  1631,  1636,  1644,
    1645,  1649,  1653,  1665,  1672,  1682,  1683,  1686,  1687,  1690,
    1692,  1696,  1703,  1704,  1705,  1717,  1716,  1775,  1778,  1784,
    1786,  1787,  1787,  1793,  1795,  1799,  1800,  1804,  1838,  1840,
    1849,  1850,  1854,  1855,  1864,  1867,  1869,  1873,  1874,  1877,
    1895,  1899,  1899,  1933,  1955,  1982,  1984,  1985,  1992,  2000,
    2006,  2012,  2026,  2025,  2089,  2090,  2096,  2098,  2102,  2103,
    2106,  2125,  2134,  2133,  2151,  2152,  2153,  2160,  2176,  2177,
    2178,  2188,  2189,  2190,  2191,  2195,  2213,  2214,  2215,  2226,
    2227,  2232,  2237,  2243,  2256,  2257,  2258,  2259,  2260,  2261,
    2262,  2263,  2264,  2265,  2266,  2267,  2268,  2269,  2270,  2271,
    2273,  2275,  2276,  2277,  2278,  2279,  2280,  2281,  2282,  2283,
    2284,  2285,  2286,  2287,  2288,  2289,  2290,  2291,  2292,  2293,
    2294,  2295,  2296,  2297,  2298,  2299,  2300,  2308,  2312,  2316,
    2320,  2321,  2322,  2323,  2324,  2329,  2332,  2335,  2338,  2344,
    2350,  2355,  2355,  2365,  2364,  2407,  2408,  2412,  2421,  2420,
    2464,  2465,  2474,  2479,  2486,  2493,  2503,  2504,  2508,  2515,
    2516,  2520,  2529,  2530,  2531,  2539,  2540,  2544,  2545,  2549,
    2556,  2563,  2564
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "END", "DEF", "WHILE", "BREAK", "CONTINUE", "DROPPING",
  "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST", "FORALL", "SWITCH",
  "CASE", "DEFAULT", "SELECT", "SENDER", "SELF", "GIVE", "TRY", "CATCH",
  "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL", "LAMBDA", "INIT",
  "LOAD", "LAUNCH", "CONST_KW", "ATTRIBUTES", "PASS", "EXPORT",
  "DIRECTIVE", "COLON", "FUNCDECL", "STATIC", "FORDOT", "LISTPAR", "LOOP",
  "OUTER_STRING", "CLOSEPAR", "OPENPAR", "CLOSESQUARE", "OPENSQUARE",
  "DOT", "ASSIGN_POW", "ASSIGN_SHL", "ASSIGN_SHR", "ASSIGN_BXOR",
  "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD", "ASSIGN_DIV", "ASSIGN_MUL",
  "ASSIGN_SUB", "ASSIGN_ADD", "ARROW", "FOR_STEP", "OP_TO", "COMMA",
  "QUESTION", "OR", "AND", "NOT", "LET", "LE", "GE", "LT", "GT", "NEQ",
  "EEQ", "OP_EQ", "OP_ASSIGN", "PROVIDES", "OP_NOTIN", "OP_IN", "HASNT",
  "HAS", "DIESIS", "ATSIGN", "CAP", "VBAR", "AMPER", "MINUS", "PLUS",
  "PERCENT", "SLASH", "STAR", "POW", "SHR", "SHL", "BANG", "NEG",
  "DECREMENT", "INCREMENT", "DOLLAR", "$accept", "input", "body", "line",
  "toplevel_statement", "load_statement", "statement",
  "assignment_def_list", "base_statement", "def_statement", "assignment",
  "op_assignment", "autoadd", "autosub", "automul", "autodiv", "automod",
  "autopow", "autoband", "autobor", "autobxor", "autoshl", "autoshr",
  "while_statement", "@1", "while_decl", "while_short_decl",
  "if_statement", "@2", "if_decl", "if_short_decl", "elif_or_else", "@3",
  "else_decl", "elif_statement", "@4", "elif_decl", "statement_list",
  "break_statement", "continue_statement", "for_statement", "@5",
  "for_decl", "for_decl_short", "forin_statement", "@6", "@7",
  "forin_statement_list", "forin_statement_elem", "fordot_statement",
  "self_print_statement", "outer_print_statement", "first_loop_block",
  "@8", "last_loop_block", "@9", "all_loop_block", "@10",
  "switch_statement", "@11", "switch_decl", "case_list", "case_statement",
  "@12", "@13", "@14", "@15", "default_statement", "@16", "default_decl",
  "default_body", "case_expression_list", "case_element",
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
  "expression_pair_list", 0
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
     355,   356,   357,   358,   359,   360,   361,   362,   363
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   109,   110,   111,   111,   112,   112,   112,   113,   113,
     113,   113,   113,   113,   113,   113,   113,   114,   114,   114,
     115,   115,   115,   115,   115,   115,   116,   116,   117,   117,
     117,   117,   117,   117,   117,   117,   117,   117,   117,   117,
     117,   117,   117,   117,   117,   117,   117,   117,   117,   117,
     118,   118,   119,   119,   119,   119,   119,   119,   120,   120,
     120,   120,   120,   120,   120,   120,   120,   120,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     133,   132,   132,   134,   134,   134,   135,   135,   135,   137,
     136,   136,   138,   138,   139,   139,   140,   140,   141,   140,
     142,   142,   144,   143,   145,   145,   146,   146,   147,   147,
     148,   148,   148,   150,   149,   149,   151,   151,   151,   152,
     152,   152,   154,   153,   155,   153,   153,   156,   156,   157,
     157,   157,   157,   158,   158,   159,   159,   159,   159,   159,
     159,   160,   162,   161,   161,   161,   164,   163,   163,   163,
     166,   165,   165,   165,   168,   167,   169,   169,   170,   170,
     170,   171,   172,   171,   173,   171,   174,   171,   175,   171,
     176,   177,   176,   178,   178,   178,   179,   179,   180,   180,
     181,   181,   181,   181,   181,   183,   182,   184,   184,   185,
     185,   185,   186,   187,   186,   188,   186,   189,   186,   190,
     186,   191,   191,   192,   192,   192,   193,   193,   193,   194,
     195,   194,   196,   196,   197,   197,   198,   198,   199,   200,
     200,   200,   200,   200,   201,   201,   202,   202,   203,   203,
     204,   204,   205,   206,   205,   205,   207,   208,   207,   209,
     210,   210,   210,   211,   212,   213,   212,   214,   212,   215,
     215,   216,   216,   217,   217,   218,   218,   218,   218,   219,
     219,   219,   220,   220,   220,   221,   221,   222,   222,   223,
     223,   224,   224,   225,   225,   226,   226,   227,   227,   228,
     228,   228,   229,   229,   229,   231,   230,   232,   232,   233,
     233,   234,   233,   235,   235,   236,   236,   237,   238,   238,
     239,   239,   240,   240,   240,   241,   241,   242,   242,   242,
     242,   244,   243,   245,   245,   246,   246,   246,   247,   247,
     247,   247,   249,   248,   250,   250,   251,   251,   252,   252,
     252,   252,   254,   253,   255,   255,   255,   256,   257,   257,
     257,   258,   258,   258,   258,   259,   260,   260,   260,   261,
     261,   261,   261,   261,   262,   262,   262,   262,   262,   262,
     262,   262,   262,   262,   262,   262,   262,   262,   262,   262,
     262,   262,   262,   262,   262,   262,   262,   262,   262,   262,
     262,   262,   262,   262,   262,   262,   262,   262,   262,   262,
     262,   262,   262,   262,   262,   262,   262,   262,   262,   262,
     262,   262,   262,   262,   262,   263,   263,   263,   263,   264,
     264,   265,   264,   267,   266,   268,   268,   268,   270,   269,
     271,   271,   272,   272,   272,   272,   273,   273,   273,   273,
     273,   273,   274,   274,   274,   275,   275,   276,   276,   277,
     277,   278,   278
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
       2,     4,     5,     3,     1,     1,     3,     2,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     2,     4,
       4,     3,     2,     2,     2,     2,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     2,     3,     3,     3,     3,
       3,     2,     2,     2,     1,     1,     3,     4,     5,     2,
       1,     1,     1,     1,     3,     3,     4,     4,     5,     4,
       3,     0,     6,     0,     6,     4,     3,     2,     0,     5,
       1,     2,     5,     5,     4,     3,     2,     3,     3,     2,
       3,     3,     3,     3,     4,     1,     3,     1,     2,     1,
       3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    21,   342,   343,   345,   344,
     341,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   348,   347,     0,     0,     0,     0,     0,     0,   332,
     418,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    84,   141,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     4,     5,     8,    13,
      20,    29,    30,    31,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    67,    68,    32,    80,     0,    37,    89,
       0,    38,    39,    33,   113,     0,    34,    47,    48,    49,
      35,   154,    36,   185,    40,    41,   210,    42,    10,   244,
       0,     0,    45,    46,    14,    15,     9,    16,     0,   279,
      11,    12,    44,    43,   354,   346,   349,   355,     0,   403,
     395,   394,   400,   401,   402,    25,     6,     0,     0,     0,
       0,   355,     0,     0,   108,     0,   110,     0,     0,     0,
       0,   346,     0,     0,     0,     0,     0,     0,     0,     0,
     435,     0,     0,   212,     0,     0,     0,     0,   285,     0,
     322,     0,   338,     0,     0,     0,     0,     0,     0,     0,
       0,   395,     0,     0,     0,   275,   277,     0,     0,     0,
     262,   265,     0,     0,     0,     0,   269,     0,   239,     0,
       0,   429,   437,     0,    87,     0,     0,   426,     0,   435,
       0,     0,   385,     0,     0,   138,     0,   393,   392,   357,
       0,   136,     0,   368,   375,   373,   391,   106,     0,     0,
       0,    82,   106,    91,   106,   115,   158,   189,   106,     0,
     106,   245,   247,   231,     0,     0,     0,   280,     0,   279,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   350,    28,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   374,   372,     0,     0,
     399,    51,    50,     0,     0,    85,    88,    83,    86,   109,
     112,   111,    93,    95,    92,    94,   118,   121,     0,     0,
       0,   157,   156,     7,   188,   187,   208,     0,     0,     0,
     213,   209,   229,   228,    24,     0,    23,     0,   340,   339,
     337,     0,   334,     0,   243,   420,   241,     0,    19,    17,
      18,   254,   253,   261,     0,   276,   278,   258,   255,     0,
     264,   263,     0,   268,     0,   267,     0,    22,   134,   133,
     431,   430,   438,   404,   405,     0,   432,     0,     0,   428,
     427,     0,   433,     0,     0,     0,   140,   137,   139,   135,
       0,     0,     0,     0,     0,     0,     0,   249,   251,     0,
     106,     0,   235,   237,     0,   284,   282,     0,     0,     0,
     274,     0,     0,   353,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   439,     0,   413,     0,   435,     0,
       0,   410,     0,     0,   425,     0,   384,   383,   382,   381,
     380,   379,   371,   376,   377,   378,   390,   389,   388,   387,
     386,   365,   364,   363,   358,   356,   361,   360,   359,   362,
     367,   366,     0,     0,   396,     0,    26,     0,   440,     0,
       0,   207,     0,   436,     0,   240,   305,   293,     0,     0,
       0,   326,   333,     0,   421,     0,     0,     0,     0,     0,
     388,   266,   272,   271,   270,   407,   406,     0,   441,   434,
       0,   369,   370,     0,   107,     0,     0,     0,    98,    97,
     102,     0,     0,   161,     0,     0,   159,     0,   171,     0,
     192,     0,     0,   190,     0,     0,   215,   216,   106,   250,
     252,     0,     0,   248,     0,   233,     0,   281,   273,   283,
       0,   351,    74,    78,    79,    77,    76,    75,    73,    72,
      71,    70,     0,     0,     0,    52,    55,    54,   411,   409,
      69,   424,     0,     0,   397,     0,     0,   126,   122,   124,
     206,   288,     0,   315,     0,   325,   298,   294,   295,   324,
     315,   336,   335,   242,   419,   260,   259,   257,   256,   408,
       0,    81,     0,   100,     0,     0,     0,   106,   106,   114,
     160,     0,   181,   184,   182,   180,     0,   178,   175,     0,
       0,   191,     0,   204,   205,     0,   201,     0,     0,   219,
     226,   227,     0,     0,   224,     0,   217,     0,   230,     0,
       0,     0,   232,   236,   352,   435,     0,     0,   240,   244,
      53,     0,   423,   422,   398,    27,     0,     0,     0,   291,
     290,   307,     0,     0,     0,     0,     0,   308,   306,   310,
     309,     0,   287,     0,   297,     0,   328,   329,   331,   330,
       0,   327,   442,   101,   105,   104,    90,     0,     0,   166,
     168,     0,   162,   164,     0,   155,   106,     0,   172,   197,
     199,   193,   195,   203,   186,   223,     0,   221,     0,     0,
     211,   246,   238,     0,    56,    57,   417,     0,   106,   412,
     116,   119,     0,     0,     0,     0,   129,     0,     0,   130,
     131,   132,   125,     0,     0,   311,     0,     0,   318,     0,
       0,     0,   303,   304,     0,   300,   302,   296,     0,   103,
     106,     0,   183,   106,     0,   179,     0,   177,   106,     0,
     106,     0,   202,   220,   225,     0,   234,   416,     0,     0,
       0,     0,   142,     0,     0,   146,     0,     0,   150,     0,
       0,   128,   292,     0,   244,     0,   317,   319,   316,     0,
     286,   299,     0,   323,     0,   169,     0,   165,     0,   200,
       0,   196,   222,   415,   414,   117,   120,   145,   106,   144,
     149,   106,   148,   153,   106,   152,   123,   314,   106,     0,
     320,     0,   301,     0,     0,     0,     0,   313,   321,     0,
       0,     0,     0,   143,   147,   151,   312
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    56,    57,    58,   494,   128,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,   217,    76,    77,    78,   222,    79,
      80,   497,   587,   498,   499,   588,   500,   380,    81,    82,
      83,   224,    84,    85,    86,   637,   638,   707,   708,    87,
      88,    89,   709,   788,   710,   791,   711,   794,    90,   226,
      91,   383,   506,   733,   734,   730,   731,   507,   600,   508,
     678,   596,   597,    92,   227,    93,   384,   513,   740,   741,
     738,   739,   605,   606,    94,    95,   228,    96,   515,   516,
     517,   518,   613,   614,    97,    98,    99,   621,   100,   524,
     101,   335,   336,   230,   390,   391,   231,   232,   102,   103,
     104,   105,   182,   106,   185,   186,   107,   108,   109,   238,
     239,   110,   325,   466,   467,   713,   470,   567,   568,   654,
     724,   725,   563,   648,   649,   764,   650,   651,   720,   111,
     327,   471,   570,   661,   112,   164,   331,   332,   113,   114,
     115,   116,   131,   118,   119,   120,   631,   419,   543,   629,
     121,   165,   337,   122,   123,   124,   151,   193,   143,   201
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -623
static const yytype_int16 yypact[] =
{
    -623,   134,   805,  -623,    15,  -623,  -623,  -623,  -623,  -623,
    -623,    47,    53,   688,   318,   460,  3025,   431,  3085,    94,
    3119,  -623,  -623,  3179,   195,  3213,   194,   250,   110,  -623,
    -623,   448,  3273,   316,   269,  3307,   243,   490,   517,  3367,
    5227,    55,  -623,  5478,  5090,  5478,   193,   344,  5478,  5478,
    5478,   438,  5478,  5478,  5478,  5478,  -623,  -623,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,  2965,  -623,  -623,
    2965,  -623,  -623,  -623,  -623,  2965,  -623,  -623,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,   188,
    2965,    19,  -623,  -623,  -623,  -623,  -623,  -623,    18,   239,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,   595,  3736,  -623,
     416,  -623,  -623,  -623,  -623,  -623,  -623,   256,    77,   222,
     311,   441,  3794,   310,  -623,   317,  -623,   325,   315,  3852,
     326,   -45,    -9,   185,   343,  4018,   364,   413,  4056,   423,
    6045,    54,   427,  -623,  2965,   435,  4106,   444,  -623,   450,
    -623,   453,  -623,  4144,   470,    52,   480,   487,   496,   514,
    6045,    83,   525,   334,   385,  -623,  -623,   528,  4194,   537,
    -623,  -623,    86,   538,   459,   132,  -623,   549,  -623,   551,
    4232,  -623,  6045,   594,  -623,  5773,  5332,  -623,   498,  5619,
      95,    98,  6156,   465,   552,  -623,   166,   308,   308,   306,
     554,  -623,   168,   306,   306,   306,   306,  -623,   561,   564,
     567,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,   329,
    -623,  -623,  -623,  -623,   568,    40,   572,  -623,   170,    30,
     171,  5115,   570,  5478,  5478,  5478,  5478,  5478,  5478,  5478,
    5478,  5478,  5478,   573,  5357,  -623,  -623,  5366,  5478,  3401,
    5478,  5478,  5478,  5478,  5478,  5478,  5478,  5478,  5478,  5478,
     577,  5478,  5478,  5478,  5478,  5478,  5478,  5478,  5478,  5478,
    5478,  5478,  5478,  5478,  5478,  5478,  -623,  -623,  5220,   580,
    -623,  -623,  -623,   573,  5478,  -623,  -623,  -623,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  5478,   573,
    3461,  -623,  -623,  -623,  -623,  -623,  -623,   575,  5478,  5478,
    -623,  -623,  -623,  -623,  -623,   323,  -623,   158,  -623,  -623,
    -623,   176,  -623,   588,  -623,   532,  -623,   534,  -623,  -623,
    -623,  -623,  -623,  -623,   559,  -623,  -623,  -623,  -623,  3495,
    -623,  -623,   590,  -623,   249,  -623,   599,  -623,  -623,  -623,
    -623,  -623,  6045,  -623,  -623,  5810,  -623,  5471,  5478,  -623,
    -623,   553,  -623,  5478,  5478,  5478,  -623,  -623,  -623,  -623,
    1777,  1453,  1885,   486,   524,  1561,   414,  -623,  -623,  1993,
    -623,  2965,  -623,  -623,    50,  -623,  -623,   601,   606,   178,
    -623,  5478,  5677,  -623,  4282,  4320,  4370,  4408,  4458,  4496,
    4546,  4584,  4634,  4672,  -623,    69,  -623,  5583,  4722,   607,
     180,  -623,    93,  4760,  -623,  3628,  6119,  6156,  5529,  5529,
    5529,  5529,  5529,  5529,  5529,  5529,  -623,   308,   308,   308,
     308,   527,   527,   405,   379,   379,   252,   252,   252,   262,
     306,   306,  5478,  5735,  -623,   526,  6045,  5847,  -623,   609,
    3910,  -623,  4810,  6045,   610,   608,  -623,   583,   614,   612,
     616,  -623,  -623,   550,  -623,   608,  5478,   629,   634,   635,
     128,  -623,  -623,  -623,  -623,  -623,  -623,  5884,  6045,  -623,
    5934,  5529,  5529,   638,  -623,   337,  3555,   633,  -623,  -623,
    -623,   640,   642,  -623,   581,   330,  -623,   637,  -623,   644,
    -623,    28,   654,  -623,    22,   655,   639,  -623,  -623,  -623,
    -623,   665,  2101,  -623,   619,  -623,   420,  -623,  -623,  -623,
    5971,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,
    -623,  -623,  5478,    37,   101,  -623,  -623,  -623,  -623,  -623,
    -623,  -623,  3589,  6008,  -623,  5478,  5478,  -623,  -623,  -623,
    -623,  -623,   122,    82,   670,  -623,   620,   604,  -623,  -623,
     142,  -623,  -623,  -623,  6045,  -623,  -623,  -623,  -623,  -623,
    5478,  -623,   674,  -623,   675,  4848,   678,  -623,  -623,  -623,
    -623,   422,   611,  -623,  -623,  -623,   116,  -623,  -623,   681,
     424,  -623,   425,  -623,  -623,   165,  -623,   684,   685,  -623,
    -623,  -623,   573,     9,  -623,   694,  -623,  1669,  -623,   696,
     645,   650,  -623,  -623,  -623,  4898,   182,   701,   608,   188,
    -623,   652,  -623,  6082,  -623,  6045,  3686,   913,  2965,  -623,
    -623,  -623,   622,   705,   703,   706,    16,  -623,  -623,  -623,
    -623,   702,  -623,   531,  -623,   612,  -623,  -623,  -623,  -623,
     704,  -623,  6045,  -623,  -623,  -623,  -623,  2209,  1453,  -623,
    -623,   712,  -623,  -623,   586,  -623,  -623,  2965,  -623,  -623,
    -623,  -623,  -623,   338,  -623,  -623,   714,  -623,   393,   573,
    -623,  -623,  -623,   715,  -623,  -623,  -623,   131,  -623,  -623,
    -623,  -623,  5478,   336,   340,   421,  -623,   711,   913,  -623,
    -623,  -623,  -623,   668,  5478,  -623,   641,   719,  -623,   717,
     236,   721,  -623,  -623,    91,  -623,  -623,  -623,   724,  -623,
    -623,  2965,  -623,  -623,  2965,  -623,  2317,  -623,  -623,  2965,
    -623,  2965,  -623,  -623,  -623,   725,  -623,  -623,   727,  2425,
    3968,   732,  -623,  2965,   737,  -623,  2965,   742,  -623,  2965,
     743,  -623,  -623,  4936,   188,  5478,  -623,  -623,  -623,    57,
    -623,  -623,   531,  -623,  1021,  -623,  1129,  -623,  1237,  -623,
    1345,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  4986,
    -623,   744,  -623,  2533,  2641,  2749,  2857,  -623,  -623,   745,
     746,   749,   750,  -623,  -623,  -623,  -623
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -623,  -623,  -623,  -623,  -623,  -623,     2,  -623,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,
    -623,    88,  -623,  -623,  -623,  -623,  -623,  -214,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,    51,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,   376,  -623,  -623,
    -623,  -623,    89,  -623,  -623,  -623,  -623,  -623,  -623,  -623,
    -623,  -623,  -623,    79,  -623,  -623,  -623,  -623,  -623,  -623,
     251,  -623,  -623,    78,  -623,  -486,  -623,  -623,  -623,  -623,
    -623,  -232,   293,  -622,  -623,  -623,  -623,  -623,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,   415,  -623,  -623,  -623,   -96,
    -623,  -623,  -623,  -623,  -623,  -623,   302,  -623,   115,  -623,
    -623,     1,  -623,  -623,   205,  -623,   207,   211,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,  -623,   312,  -623,  -343,
      -6,  -623,    -2,    17,   192,   751,  -623,  -623,  -623,  -623,
    -623,  -623,  -623,  -623,  -623,  -623,   -42,  -623,   533,  -623
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -440
static const yytype_int16 yytable[] =
{
     117,   478,   200,   394,    59,   206,   129,   698,   381,   212,
     382,   141,   687,   240,   385,   142,   389,   717,   125,   236,
     234,  -279,   718,   608,   237,   609,   610,  -439,   611,   602,
     132,  -203,   603,   139,   604,   145,   237,   148,   627,   398,
     150,   393,   156,  -439,   203,   163,   334,   241,   242,   170,
     126,   525,   178,   333,   127,   317,   190,   192,   334,     8,
     195,   199,   202,   800,   150,   207,   208,   209,   150,   213,
     214,   215,   216,   235,  -203,   117,   308,   647,   117,   221,
     292,   688,   223,   117,   657,   641,   342,   225,   642,   351,
    -279,   628,   719,  -240,   548,   146,   369,   689,   117,   371,
    -203,   194,   233,   526,   630,     6,     7,     8,     9,    10,
     612,   161,  -240,   162,     6,     7,     8,     9,    10,   672,
     643,  -240,   475,   639,  -240,   318,   319,    21,    22,   644,
     645,   578,   747,   801,     3,   355,    21,    22,    30,   288,
     289,   309,   798,   399,   771,   656,   549,    30,   642,   293,
     370,    40,   117,   372,   542,    43,   321,    44,   352,   468,
      40,  -293,   673,   772,    43,   319,    44,   319,   681,   377,
     373,   379,   646,   396,   400,   640,   522,    45,    46,   472,
     643,   529,   257,   547,   748,   695,    45,    46,   674,   644,
     645,   469,    48,    49,   475,   157,   152,    50,   153,     8,
     158,    48,    49,   475,   356,    52,    50,    53,    54,    55,
     362,   682,   420,   365,    52,   422,    53,    54,    55,    21,
      22,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   646,   562,   286,   287,   229,   683,   319,   768,
     319,   154,   397,   397,   179,   237,   180,   414,   473,   181,
     397,   159,   319,   482,   319,   483,   160,   309,   402,   291,
     404,   405,   406,   407,   408,   409,   410,   411,   412,   413,
     174,   418,   175,   310,   150,   423,   425,   426,   427,   428,
     429,   430,   431,   432,   433,   434,   435,   455,   437,   438,
     439,   440,   441,   442,   443,   444,   445,   446,   447,   448,
     449,   450,   451,   458,   617,   453,   257,   294,   769,   255,
     726,   456,   290,   299,   295,   176,   257,   172,   302,   133,
     300,   134,   173,   255,   464,   457,  -289,   460,   301,   306,
     386,   598,   387,  -174,   255,   462,   463,   751,   582,   752,
     583,   754,   603,   755,   604,   204,   311,   205,     6,     7,
       8,     9,    10,   283,   284,   285,  -289,   296,   286,   287,
     257,   303,   257,   290,   284,   285,   480,   313,   286,   287,
      21,    22,   307,   667,   668,   388,  -174,   465,   117,   117,
     117,    30,   753,   117,   487,   488,   756,   117,   345,   117,
     490,   491,   492,   523,    40,   255,   697,   610,    43,   611,
      44,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   286,   287,   314,   519,   530,   344,
      45,    46,   757,   622,   758,   669,   316,   676,   679,   726,
     320,   346,   140,   257,   216,    48,    49,     8,   322,   210,
      50,   211,     6,     7,     8,     9,    10,   324,    52,   166,
      53,    54,    55,   326,   167,   168,   328,    21,    22,   257,
     520,   135,   736,   136,    21,    22,   623,   759,   670,   553,
     677,   680,   288,   289,   137,    30,   330,   280,   281,   282,
     283,   284,   285,   338,   749,   286,   287,   502,    40,   503,
     339,   183,    43,   574,    44,  -170,   184,   241,   242,   340,
     626,   278,   279,   280,   281,   282,   283,   284,   285,   504,
     505,   286,   287,   585,    45,    46,   774,   341,   187,   776,
     117,   241,   242,   188,   778,   509,   780,   510,   343,    48,
      49,   347,  -173,  -170,    50,     6,     7,   722,     9,    10,
     350,   353,    52,   354,    53,    54,    55,   511,   505,   374,
     375,   571,   357,   366,   358,   376,   330,   378,   723,   625,
     477,   216,   157,     6,     7,   159,     9,    10,   187,   633,
    -173,   392,   635,   636,   803,   395,   403,   804,   461,     8,
     805,   257,   591,   436,   806,   592,   454,   593,   594,   595,
     592,   474,   593,   594,   595,   360,   481,   662,     6,     7,
       8,     9,    10,   476,   475,   184,   686,   527,   489,   528,
     546,   555,   557,   561,   334,   117,   469,   565,   566,   569,
      21,    22,   277,   278,   279,   280,   281,   282,   283,   284,
     285,    30,   575,   286,   287,   117,   117,   576,   577,   706,
     712,   581,   586,   589,    40,   590,   599,   601,    43,   361,
      44,   241,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   607,   615,   117,   117,   253,   618,   514,
      45,    46,   620,   652,   653,   117,   655,   663,   664,   737,
     254,   666,   671,   745,   675,    48,    49,   684,   685,   130,
      50,   692,     6,     7,     8,     9,    10,   690,    52,   691,
      53,    54,    55,   693,   696,   699,   117,   714,   715,   188,
     706,   721,   716,   728,    21,    22,   732,   743,   746,   750,
     760,   762,   766,   767,   770,    30,   765,   773,   782,   117,
     783,   763,   117,   775,   117,   787,   777,   117,    40,   117,
     790,   779,    43,   781,    44,   793,   796,   117,   813,   814,
     808,   117,   815,   816,   117,   789,   729,   117,   792,   761,
     512,   795,   742,   735,    45,    46,   744,   616,   573,   564,
     727,   484,   117,   802,   117,   658,   117,   659,   117,    48,
      49,   660,   799,   171,    50,   572,   415,     0,     0,     0,
       0,     0,    52,     0,    53,    54,    55,     0,     0,     0,
       0,   117,   117,   117,   117,    -2,     4,     0,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,    19,     0,
      20,    21,    22,    23,    24,     0,    25,    26,     0,    27,
      28,    29,    30,     0,    31,    32,    33,    34,    35,    36,
      37,     0,    38,     0,    39,    40,    41,    42,     0,    43,
       0,    44,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    45,    46,     0,     0,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,    51,     0,    52,
       0,    53,    54,    55,     4,     0,     5,     6,     7,     8,
       9,    10,  -127,    12,    13,    14,    15,     0,    16,     0,
       0,    17,   703,   704,   705,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   218,     0,   219,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
     220,     0,    39,    40,    41,    42,     0,    43,     0,    44,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    45,
      46,     0,     0,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,    51,     0,    52,     0,    53,
      54,    55,     4,     0,     5,     6,     7,     8,     9,    10,
    -167,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,  -167,  -167,    20,    21,    22,    23,
      24,     0,    25,   218,     0,   219,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,  -167,   220,     0,
      39,    40,    41,    42,     0,    43,     0,    44,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    45,    46,     0,
       0,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,    50,     0,     0,
       0,     0,     0,    51,     0,    52,     0,    53,    54,    55,
       4,     0,     5,     6,     7,     8,     9,    10,  -163,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,  -163,  -163,    20,    21,    22,    23,    24,     0,
      25,   218,     0,   219,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,  -163,   220,     0,    39,    40,
      41,    42,     0,    43,     0,    44,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    45,    46,     0,     0,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,    50,     0,     0,     0,     0,
       0,    51,     0,    52,     0,    53,    54,    55,     4,     0,
       5,     6,     7,     8,     9,    10,  -198,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
    -198,  -198,    20,    21,    22,    23,    24,     0,    25,   218,
       0,   219,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,  -198,   220,     0,    39,    40,    41,    42,
       0,    43,     0,    44,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    45,    46,     0,     0,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,    50,     0,     0,     0,     0,     0,    51,
       0,    52,     0,    53,    54,    55,     4,     0,     5,     6,
       7,     8,     9,    10,  -194,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -194,  -194,
      20,    21,    22,    23,    24,     0,    25,   218,     0,   219,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,  -194,   220,     0,    39,    40,    41,    42,     0,    43,
       0,    44,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    45,    46,     0,     0,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,    51,     0,    52,
       0,    53,    54,    55,     4,     0,     5,     6,     7,     8,
       9,    10,   -96,    12,    13,    14,    15,     0,    16,   495,
     496,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   218,     0,   219,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
     220,     0,    39,    40,    41,    42,     0,    43,     0,    44,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    45,
      46,     0,     0,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,    51,     0,    52,     0,    53,
      54,    55,     4,     0,     5,     6,     7,     8,     9,    10,
    -214,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,   514,    25,   218,     0,   219,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,   220,     0,
      39,    40,    41,    42,     0,    43,     0,    44,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    45,    46,     0,
       0,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,    50,     0,     0,
       0,     0,     0,    51,     0,    52,     0,    53,    54,    55,
       4,     0,     5,     6,     7,     8,     9,    10,  -218,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,  -218,
      25,   218,     0,   219,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   220,     0,    39,    40,
      41,    42,     0,    43,     0,    44,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    45,    46,     0,     0,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,    50,     0,     0,     0,     0,
       0,    51,     0,    52,     0,    53,    54,    55,     4,     0,
       5,     6,     7,     8,     9,    10,   493,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   218,
       0,   219,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,   220,     0,    39,    40,    41,    42,
       0,    43,     0,    44,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    45,    46,     0,     0,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,    50,     0,     0,     0,     0,     0,    51,
       0,    52,     0,    53,    54,    55,     4,     0,     5,     6,
       7,     8,     9,    10,   501,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   218,     0,   219,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,   220,     0,    39,    40,    41,    42,     0,    43,
       0,    44,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    45,    46,     0,     0,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,    51,     0,    52,
       0,    53,    54,    55,     4,     0,     5,     6,     7,     8,
       9,    10,   521,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   218,     0,   219,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
     220,     0,    39,    40,    41,    42,     0,    43,     0,    44,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    45,
      46,     0,     0,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,    51,     0,    52,     0,    53,
      54,    55,     4,     0,     5,     6,     7,     8,     9,    10,
     619,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   218,     0,   219,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,   220,     0,
      39,    40,    41,    42,     0,    43,     0,    44,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    45,    46,     0,
       0,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,    50,     0,     0,
       0,     0,     0,    51,     0,    52,     0,    53,    54,    55,
       4,     0,     5,     6,     7,     8,     9,    10,   -99,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   218,     0,   219,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   220,     0,    39,    40,
      41,    42,     0,    43,     0,    44,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    45,    46,     0,     0,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,    50,     0,     0,     0,     0,
       0,    51,     0,    52,     0,    53,    54,    55,     4,     0,
       5,     6,     7,     8,     9,    10,  -176,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   218,
       0,   219,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,   220,     0,    39,    40,    41,    42,
       0,    43,     0,    44,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    45,    46,     0,     0,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,    50,     0,     0,     0,     0,     0,    51,
       0,    52,     0,    53,    54,    55,     4,     0,     5,     6,
       7,     8,     9,    10,   784,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   218,     0,   219,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,   220,     0,    39,    40,    41,    42,     0,    43,
       0,    44,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    45,    46,     0,     0,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,    51,     0,    52,
       0,    53,    54,    55,     4,     0,     5,     6,     7,     8,
       9,    10,   809,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   218,     0,   219,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
     220,     0,    39,    40,    41,    42,     0,    43,     0,    44,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    45,
      46,     0,     0,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,    51,     0,    52,     0,    53,
      54,    55,     4,     0,     5,     6,     7,     8,     9,    10,
     810,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   218,     0,   219,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,   220,     0,
      39,    40,    41,    42,     0,    43,     0,    44,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    45,    46,     0,
       0,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,    50,     0,     0,
       0,     0,     0,    51,     0,    52,     0,    53,    54,    55,
       4,     0,     5,     6,     7,     8,     9,    10,   811,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   218,     0,   219,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,   220,     0,    39,    40,
      41,    42,     0,    43,     0,    44,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    45,    46,     0,     0,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,    50,     0,     0,     0,     0,
       0,    51,     0,    52,     0,    53,    54,    55,     4,     0,
       5,     6,     7,     8,     9,    10,   812,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   218,
       0,   219,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,   220,     0,    39,    40,    41,    42,
       0,    43,     0,    44,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    45,    46,     0,     0,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,    50,     0,     0,     0,     0,     0,    51,
       0,    52,     0,    53,    54,    55,     4,     0,     5,     6,
       7,     8,     9,    10,     0,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   218,     0,   219,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,   220,     0,    39,    40,    41,    42,     0,    43,
       0,    44,     0,     0,     0,     0,   138,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,    45,    46,     0,     0,     0,    47,     0,     0,     0,
       0,    21,    22,     0,     0,     0,    48,    49,     0,     0,
       0,    50,    30,     0,     0,     0,     0,    51,     0,    52,
       0,    53,    54,    55,     0,    40,     0,     0,     0,    43,
       0,    44,     0,     0,     0,     0,   144,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,    45,    46,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,    48,    49,     0,     0,
     147,    50,    30,     6,     7,     8,     9,    10,     0,    52,
       0,    53,    54,    55,     0,    40,     0,     0,     0,    43,
       0,    44,     0,     0,     0,    21,    22,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    30,     0,     0,     0,
       0,    45,    46,     0,     0,     0,     0,     0,     0,    40,
       0,     0,     0,    43,     0,    44,    48,    49,     0,     0,
     149,    50,     0,     6,     7,     8,     9,    10,     0,    52,
       0,    53,    54,    55,     0,    45,    46,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
      48,    49,     0,     0,   155,    50,    30,     6,     7,     8,
       9,    10,     0,    52,     0,    53,    54,    55,     0,    40,
       0,     0,     0,    43,     0,    44,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      30,     0,     0,     0,     0,    45,    46,     0,     0,     0,
       0,     0,     0,    40,     0,     0,     0,    43,     0,    44,
      48,    49,     0,     0,   169,    50,     0,     6,     7,     8,
       9,    10,     0,    52,     0,    53,    54,    55,     0,    45,
      46,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,    48,    49,     0,     0,   177,    50,
      30,     6,     7,     8,     9,    10,     0,    52,     0,    53,
      54,    55,     0,    40,     0,     0,     0,    43,     0,    44,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    30,     0,     0,     0,     0,    45,
      46,     0,     0,     0,     0,     0,     0,    40,     0,     0,
       0,    43,     0,    44,    48,    49,     0,     0,   189,    50,
       0,     6,     7,     8,     9,    10,     0,    52,     0,    53,
      54,    55,     0,    45,    46,     0,     0,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,    48,    49,
       0,     0,   424,    50,    30,     6,     7,     8,     9,    10,
       0,    52,     0,    53,    54,    55,     0,    40,     0,     0,
       0,    43,     0,    44,     0,     0,     0,    21,    22,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    30,     0,
       0,     0,     0,    45,    46,     0,     0,     0,     0,     0,
       0,    40,     0,     0,     0,    43,     0,    44,    48,    49,
       0,     0,   459,    50,     0,     6,     7,     8,     9,    10,
       0,    52,     0,    53,    54,    55,     0,    45,    46,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,     0,    48,    49,     0,     0,   479,    50,    30,     6,
       7,     8,     9,    10,     0,    52,     0,    53,    54,    55,
       0,    40,     0,     0,     0,    43,     0,    44,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,    45,    46,     0,
       0,     0,     0,     0,     0,    40,     0,     0,     0,    43,
       0,    44,    48,    49,     0,     0,   584,    50,     0,     6,
       7,     8,     9,    10,     0,    52,     0,    53,    54,    55,
       0,    45,    46,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,    48,    49,     0,     0,
     632,    50,    30,     6,     7,     8,     9,    10,     0,    52,
       0,    53,    54,    55,     0,    40,     0,     0,     0,    43,
       0,    44,     0,     0,     0,    21,    22,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    30,     0,     0,   551,
       0,    45,    46,     0,     0,     0,     0,     0,     0,    40,
       0,     0,     0,    43,     0,    44,    48,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,    52,
       0,    53,    54,    55,     0,    45,    46,     0,     0,     0,
       0,     0,     0,     0,   552,     0,     0,     0,     0,     0,
      48,    49,   257,     0,     0,    50,     0,     0,     0,   700,
       0,     0,     0,    52,     0,    53,    54,    55,     0,     0,
       0,   259,   260,   261,     0,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,     0,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   701,     0,   286,   287,     0,     0,     0,   256,
     257,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   702,     0,     0,   259,
     260,   261,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,     0,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
     257,     0,   286,   287,     0,     0,     0,   297,     0,     0,
       0,     0,     0,     0,   258,     0,     0,     0,     0,   259,
     260,   261,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,     0,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
     298,     0,   286,   287,     0,     0,     0,     0,   257,     0,
       0,     0,     0,     0,     0,   304,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   259,   260,   261,
       0,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,     0,     0,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   305,     0,
     286,   287,     0,     0,     0,     0,   257,     0,     0,     0,
       0,     0,     0,   558,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   259,   260,   261,     0,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,     0,     0,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   559,     0,   286,   287,
       0,     0,     0,     0,   257,     0,     0,     0,     0,     0,
       0,   785,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   259,   260,   261,     0,     0,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,     0,     0,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   786,     0,   286,   287,     0,     0,
       0,   312,   257,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   259,   260,   261,     0,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   315,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   257,     0,   286,   287,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   259,   260,   261,     0,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   323,
     257,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,     0,     0,   286,   287,     0,     0,     0,   259,
     260,   261,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   329,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
     257,     0,   286,   287,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   259,
     260,   261,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   348,   257,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
       0,     0,   286,   287,     0,     0,     0,   259,   260,   261,
       0,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   359,     0,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   257,     0,
     286,   287,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   259,   260,   261,
       0,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   349,   273,   274,   532,   257,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,     0,     0,
     286,   287,     0,     0,     0,   259,   260,   261,     0,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   533,     0,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   257,     0,   286,   287,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   259,   260,   261,     0,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   534,   257,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,     0,     0,   286,   287,
       0,     0,     0,   259,   260,   261,     0,     0,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   535,     0,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   257,     0,   286,   287,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   259,   260,   261,     0,     0,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   536,   257,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,     0,     0,   286,   287,     0,     0,
       0,   259,   260,   261,     0,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   537,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   257,     0,   286,   287,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   259,   260,   261,     0,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   538,
     257,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,     0,     0,   286,   287,     0,     0,     0,   259,
     260,   261,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   539,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
     257,     0,   286,   287,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   259,
     260,   261,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   540,   257,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
       0,     0,   286,   287,     0,     0,     0,   259,   260,   261,
       0,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   541,     0,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   257,     0,
     286,   287,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   259,   260,   261,
       0,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   545,   257,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,     0,     0,
     286,   287,     0,     0,     0,   259,   260,   261,     0,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   550,     0,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   257,     0,   286,   287,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   259,   260,   261,     0,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   560,   257,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,     0,     0,   286,   287,
       0,     0,     0,   259,   260,   261,     0,     0,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   665,     0,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   257,     0,   286,   287,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   259,   260,   261,     0,     0,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   694,   257,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,     0,     0,   286,   287,     0,     0,
       0,   259,   260,   261,     0,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   797,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   257,     0,   286,   287,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   259,   260,   261,     0,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   807,
     257,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,     0,     0,   286,   287,     0,     0,     0,   259,
     260,   261,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,     0,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
     257,     0,   286,   287,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   259,
     260,   261,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,     0,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
       0,     0,   286,   287,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     6,
       7,     8,     9,    10,     0,     0,     0,    30,     0,     0,
       0,     0,     0,     0,     0,     0,   196,     0,     0,     0,
      40,    21,    22,     0,    43,   197,    44,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,     0,   198,
       0,   196,     0,     0,     0,    40,    45,    46,     0,    43,
       0,    44,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,    49,     0,     0,     0,    50,     0,     0,     0,
       0,    45,    46,     0,    52,     0,    53,    54,    55,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,    50,     0,     0,     0,   401,     0,     0,     0,    52,
       0,    53,    54,    55,     6,     7,     8,     9,    10,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,    21,    22,     0,     0,    30,     0,     0,
       0,     0,     0,     0,    30,     0,   196,     0,     0,     0,
      40,     0,     0,     0,    43,     0,    44,    40,     0,     0,
       0,    43,   191,    44,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    45,    46,     0,     0,
       0,     0,     0,    45,    46,     0,     0,     0,     0,     0,
       0,    48,    49,     0,     0,     0,    50,     0,    48,    49,
     452,     0,     0,    50,    52,     0,    53,    54,    55,     0,
       0,    52,     0,    53,    54,    55,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     6,     7,     8,     9,    10,     0,     0,     0,    30,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    40,    21,    22,     0,    43,   364,    44,     0,
       0,     0,    21,    22,    30,     0,     0,     0,     0,     0,
       0,     0,     0,    30,   416,     0,     0,    40,    45,    46,
       0,    43,     0,    44,     0,     0,    40,     0,     0,   421,
      43,     0,    44,    48,    49,     0,     0,     0,    50,     0,
       0,     0,     0,    45,    46,     0,    52,     0,    53,    54,
      55,     0,    45,    46,     0,     0,     0,     0,    48,    49,
       0,     0,     0,    50,     0,     0,     0,    48,    49,     0,
       0,    52,    50,    53,    54,   417,     0,     0,     0,     0,
      52,     0,    53,    54,    55,     6,     7,     8,     9,    10,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,     0,     0,     0,    21,    22,     0,     0,    30,     0,
       0,     0,     0,     0,     0,    30,     0,     0,     0,     0,
       0,    40,     0,     0,     0,    43,   486,    44,    40,     0,
       0,     0,    43,     0,    44,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    45,    46,     0,
       0,     0,     0,     0,    45,    46,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,    50,     0,    48,
      49,     0,     0,     0,    50,    52,     0,    53,    54,    55,
       0,     0,    52,   257,    53,    54,    55,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,   270,   271,   272,   273,   274,
      30,     0,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,    40,     0,   286,   287,    43,     0,    44,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    45,
      46,     0,     0,     0,     0,   367,     0,     0,     0,     0,
       0,     0,     0,   257,    48,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,    52,   368,    53,
      54,   544,   259,   260,   261,     0,     0,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
       0,     0,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   367,     0,   286,   287,     0,     0,     0,
       0,   257,   531,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     259,   260,   261,     0,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,     0,     0,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   367,     0,   286,   287,     0,     0,     0,     0,   257,
     554,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   259,   260,
     261,     0,     0,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   363,   257,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   285,     0,
       0,   286,   287,     0,     0,     0,   259,   260,   261,     0,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   257,   485,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,   285,     0,     0,   286,
     287,     0,     0,   259,   260,   261,     0,     0,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   257,     0,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,     0,     0,   286,   287,   556,     0,
     259,   260,   261,     0,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   257,   579,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,     0,     0,   286,   287,     0,     0,   259,   260,   261,
       0,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,     0,     0,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   257,     0,
     286,   287,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   580,     0,     0,     0,   259,   260,   261,
       0,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   257,   624,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,     0,     0,
     286,   287,     0,     0,   259,   260,   261,     0,     0,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   257,   634,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,     0,     0,   286,   287,     0,
       0,   259,   260,   261,     0,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   257,
       0,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,     0,     0,   286,   287,     0,     0,   259,   260,
     261,     0,     0,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   257,     0,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   285,     0,
       0,   286,   287,     0,     0,     0,   260,   261,     0,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   257,     0,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,     0,     0,   286,   287,
       0,     0,     0,     0,   261,     0,     0,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     257,     0,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,     0,     0,   286,   287,     0,     0,     0,
       0,     0,     0,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,     0,     0,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
       0,     0,   286,   287
};

static const yytype_int16 yycheck[] =
{
       2,   344,    44,   235,     2,    47,    12,   629,   222,    51,
     224,    17,     3,   109,   228,    17,   230,     1,     3,     1,
       1,     3,     6,     1,     6,     3,     4,    72,     6,     1,
      13,     3,     4,    16,     6,    18,     6,    20,     1,     9,
      23,     1,    25,    88,    46,    28,     6,    56,    57,    32,
       3,     1,    35,     1,     1,     1,    39,    40,     6,     6,
      43,    44,    45,     6,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    54,    46,    77,    85,   563,    80,    77,
       3,    72,    80,    85,   570,     3,     3,    85,     6,     3,
      72,    54,    76,    53,     1,     1,     1,    88,   100,     1,
      72,    46,   100,    53,     3,     4,     5,     6,     7,     8,
      88,     1,    72,     3,     4,     5,     6,     7,     8,     3,
      38,    69,    72,     1,    72,    71,    72,    26,    27,    47,
      48,     3,     1,    76,     0,     3,    26,    27,    37,    56,
      57,    72,   764,   239,    53,     3,    53,    37,     6,    72,
      55,    50,   154,    55,    85,    54,   154,    56,    72,     1,
      50,     3,    46,    72,    54,    72,    56,    72,     3,     3,
      72,     3,    90,     3,     3,    53,   390,    76,    77,     3,
      38,     3,    54,     3,    53,     3,    76,    77,    72,    47,
      48,    33,    91,    92,    72,     1,     1,    96,     3,     6,
       6,    91,    92,    72,    72,   104,    96,   106,   107,   108,
     193,    46,   254,   196,   104,   257,   106,   107,   108,    26,
      27,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    90,   465,   106,   107,    48,    72,    72,     3,
      72,    46,    72,    72,     1,     6,     3,   253,    72,     6,
      72,     1,    72,     4,    72,     6,     6,    72,   241,     3,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
       1,   254,     3,    88,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   293,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   309,   518,   288,    54,    85,    72,   117,
     653,   294,   120,     3,     3,    46,    54,     1,     3,     1,
       3,     3,     6,   131,     1,   308,     3,   310,     3,     3,
       1,     1,     3,     3,   142,   318,   319,     1,     1,     3,
       3,     1,     4,     3,     6,     1,     3,     3,     4,     5,
       6,     7,     8,   101,   102,   103,    33,    46,   106,   107,
      54,    46,    54,   171,   102,   103,   349,     3,   106,   107,
      26,    27,    46,   587,   588,    46,    46,    54,   380,   381,
     382,    37,    46,   385,   367,   368,    46,   389,     3,   391,
     373,   374,   375,   391,    50,   203,   628,     4,    54,     6,
      56,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   106,   107,   106,   107,     3,     3,   401,    85,
      76,    77,     1,     3,     3,     3,     3,     3,     3,   772,
       3,    46,     1,    54,   417,    91,    92,     6,     3,     1,
      96,     3,     4,     5,     6,     7,     8,     3,   104,     1,
     106,   107,   108,     3,     6,     7,     3,    26,    27,    54,
      46,     1,   676,     3,    26,    27,    46,    46,    46,   452,
      46,    46,    56,    57,    14,    37,     6,    98,    99,   100,
     101,   102,   103,     3,   698,   106,   107,     1,    50,     3,
       3,     1,    54,   476,    56,     9,     6,    56,    57,     3,
     542,    96,    97,    98,    99,   100,   101,   102,   103,    23,
      24,   106,   107,   496,    76,    77,   730,     3,     1,   733,
     522,    56,    57,     6,   738,     1,   740,     3,     3,    91,
      92,     3,    46,     9,    96,     4,     5,     6,     7,     8,
       3,     3,   104,    84,   106,   107,   108,    23,    24,    84,
      85,     1,     3,    55,     3,     3,     6,     3,    27,   542,
       1,   544,     1,     4,     5,     1,     7,     8,     1,   552,
      46,     3,   555,   556,   788,     3,     6,   791,     3,     6,
     794,    54,     1,     6,   798,     4,     6,     6,     7,     8,
       4,     3,     6,     7,     8,     1,     6,   580,     4,     5,
       6,     7,     8,    69,    72,     6,   612,     6,    55,     3,
       3,    85,     3,     3,     6,   617,    33,     3,     6,     3,
      26,    27,    95,    96,    97,    98,    99,   100,   101,   102,
     103,    37,     3,   106,   107,   637,   638,     3,     3,   637,
     638,     3,     9,     3,    50,     3,     9,     3,    54,    55,
      56,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,     9,     9,   667,   668,    72,     3,    30,
      76,    77,    53,     3,    54,   677,    72,     3,     3,   677,
      85,     3,    71,   689,     3,    91,    92,     3,     3,     1,
      96,    46,     4,     5,     6,     7,     8,     3,   104,     3,
     106,   107,   108,    53,     3,    53,   708,    85,     3,     6,
     708,     9,     6,     9,    26,    27,     4,     3,     3,   702,
       9,    53,     3,     6,     3,    37,    85,     3,     3,   731,
       3,   714,   734,   731,   736,     3,   734,   739,    50,   741,
       3,   739,    54,   741,    56,     3,     3,   749,     3,     3,
       6,   753,     3,     3,   756,   753,   668,   759,   756,   708,
     384,   759,   683,   674,    76,    77,   688,   516,   475,   467,
     655,   356,   774,   772,   776,   570,   778,   570,   780,    91,
      92,   570,   765,    32,    96,   473,   253,    -1,    -1,    -1,
      -1,    -1,   104,    -1,   106,   107,   108,    -1,    -1,    -1,
      -1,   803,   804,   805,   806,     0,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    39,    40,    41,    42,    43,    44,
      45,    -1,    47,    -1,    49,    50,    51,    52,    -1,    54,
      -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    76,    77,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,
      -1,    96,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
      -1,   106,   107,   108,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    19,    20,    21,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      47,    -1,    49,    50,    51,    52,    -1,    54,    -1,    56,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,    -1,   106,
     107,   108,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    23,    24,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    46,    47,    -1,
      49,    50,    51,    52,    -1,    54,    -1,    56,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    91,    92,    -1,    -1,    -1,    96,    -1,    -1,
      -1,    -1,    -1,   102,    -1,   104,    -1,   106,   107,   108,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    46,    47,    -1,    49,    50,
      51,    52,    -1,    54,    -1,    56,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    77,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      91,    92,    -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,
      -1,   102,    -1,   104,    -1,   106,   107,   108,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    46,    47,    -1,    49,    50,    51,    52,
      -1,    54,    -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    76,    77,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,    92,
      -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,    -1,   106,   107,   108,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    46,    47,    -1,    49,    50,    51,    52,    -1,    54,
      -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    76,    77,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,
      -1,    96,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
      -1,   106,   107,   108,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    16,
      17,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      47,    -1,    49,    50,    51,    52,    -1,    54,    -1,    56,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,    -1,   106,
     107,   108,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    30,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    47,    -1,
      49,    50,    51,    52,    -1,    54,    -1,    56,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    91,    92,    -1,    -1,    -1,    96,    -1,    -1,
      -1,    -1,    -1,   102,    -1,   104,    -1,   106,   107,   108,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    30,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    -1,    54,    -1,    56,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    77,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      91,    92,    -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,
      -1,   102,    -1,   104,    -1,   106,   107,   108,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    47,    -1,    49,    50,    51,    52,
      -1,    54,    -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    76,    77,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,    92,
      -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,    -1,   106,   107,   108,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    47,    -1,    49,    50,    51,    52,    -1,    54,
      -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    76,    77,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,
      -1,    96,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
      -1,   106,   107,   108,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      47,    -1,    49,    50,    51,    52,    -1,    54,    -1,    56,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,    -1,   106,
     107,   108,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    47,    -1,
      49,    50,    51,    52,    -1,    54,    -1,    56,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    91,    92,    -1,    -1,    -1,    96,    -1,    -1,
      -1,    -1,    -1,   102,    -1,   104,    -1,   106,   107,   108,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    -1,    54,    -1,    56,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    77,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      91,    92,    -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,
      -1,   102,    -1,   104,    -1,   106,   107,   108,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    47,    -1,    49,    50,    51,    52,
      -1,    54,    -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    76,    77,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,    92,
      -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,    -1,   106,   107,   108,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    47,    -1,    49,    50,    51,    52,    -1,    54,
      -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    76,    77,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,
      -1,    96,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,
      -1,   106,   107,   108,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      47,    -1,    49,    50,    51,    52,    -1,    54,    -1,    56,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,
      -1,    -1,    -1,    -1,    -1,   102,    -1,   104,    -1,   106,
     107,   108,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    47,    -1,
      49,    50,    51,    52,    -1,    54,    -1,    56,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    91,    92,    -1,    -1,    -1,    96,    -1,    -1,
      -1,    -1,    -1,   102,    -1,   104,    -1,   106,   107,   108,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    -1,    49,    50,
      51,    52,    -1,    54,    -1,    56,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    76,    77,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      91,    92,    -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,
      -1,   102,    -1,   104,    -1,   106,   107,   108,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    47,    -1,    49,    50,    51,    52,
      -1,    54,    -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    76,    77,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,    92,
      -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,   102,
      -1,   104,    -1,   106,   107,   108,     1,    -1,     3,     4,
       5,     6,     7,     8,    -1,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    47,    -1,    49,    50,    51,    52,    -1,    54,
      -1,    56,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    76,    77,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    91,    92,    -1,    -1,
      -1,    96,    37,    -1,    -1,    -1,    -1,   102,    -1,   104,
      -1,   106,   107,   108,    -1,    50,    -1,    -1,    -1,    54,
      -1,    56,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    76,    77,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    91,    92,    -1,    -1,
       1,    96,    37,     4,     5,     6,     7,     8,    -1,   104,
      -1,   106,   107,   108,    -1,    50,    -1,    -1,    -1,    54,
      -1,    56,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,
      -1,    76,    77,    -1,    -1,    -1,    -1,    -1,    -1,    50,
      -1,    -1,    -1,    54,    -1,    56,    91,    92,    -1,    -1,
       1,    96,    -1,     4,     5,     6,     7,     8,    -1,   104,
      -1,   106,   107,   108,    -1,    76,    77,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      91,    92,    -1,    -1,     1,    96,    37,     4,     5,     6,
       7,     8,    -1,   104,    -1,   106,   107,   108,    -1,    50,
      -1,    -1,    -1,    54,    -1,    56,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      37,    -1,    -1,    -1,    -1,    76,    77,    -1,    -1,    -1,
      -1,    -1,    -1,    50,    -1,    -1,    -1,    54,    -1,    56,
      91,    92,    -1,    -1,     1,    96,    -1,     4,     5,     6,
       7,     8,    -1,   104,    -1,   106,   107,   108,    -1,    76,
      77,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    91,    92,    -1,    -1,     1,    96,
      37,     4,     5,     6,     7,     8,    -1,   104,    -1,   106,
     107,   108,    -1,    50,    -1,    -1,    -1,    54,    -1,    56,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    76,
      77,    -1,    -1,    -1,    -1,    -1,    -1,    50,    -1,    -1,
      -1,    54,    -1,    56,    91,    92,    -1,    -1,     1,    96,
      -1,     4,     5,     6,     7,     8,    -1,   104,    -1,   106,
     107,   108,    -1,    76,    77,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    91,    92,
      -1,    -1,     1,    96,    37,     4,     5,     6,     7,     8,
      -1,   104,    -1,   106,   107,   108,    -1,    50,    -1,    -1,
      -1,    54,    -1,    56,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,
      -1,    -1,    -1,    76,    77,    -1,    -1,    -1,    -1,    -1,
      -1,    50,    -1,    -1,    -1,    54,    -1,    56,    91,    92,
      -1,    -1,     1,    96,    -1,     4,     5,     6,     7,     8,
      -1,   104,    -1,   106,   107,   108,    -1,    76,    77,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    91,    92,    -1,    -1,     1,    96,    37,     4,
       5,     6,     7,     8,    -1,   104,    -1,   106,   107,   108,
      -1,    50,    -1,    -1,    -1,    54,    -1,    56,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    76,    77,    -1,
      -1,    -1,    -1,    -1,    -1,    50,    -1,    -1,    -1,    54,
      -1,    56,    91,    92,    -1,    -1,     1,    96,    -1,     4,
       5,     6,     7,     8,    -1,   104,    -1,   106,   107,   108,
      -1,    76,    77,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    91,    92,    -1,    -1,
       1,    96,    37,     4,     5,     6,     7,     8,    -1,   104,
      -1,   106,   107,   108,    -1,    50,    -1,    -1,    -1,    54,
      -1,    56,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,     1,
      -1,    76,    77,    -1,    -1,    -1,    -1,    -1,    -1,    50,
      -1,    -1,    -1,    54,    -1,    56,    91,    92,    -1,    -1,
      -1,    96,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,   107,   108,    -1,    76,    77,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    46,    -1,    -1,    -1,    -1,    -1,
      91,    92,    54,    -1,    -1,    96,    -1,    -1,    -1,     3,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,    -1,    -1,
      -1,    73,    74,    75,    -1,    -1,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    -1,
      -1,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    46,    -1,   106,   107,    -1,    -1,    -1,     3,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    -1,    -1,    73,
      74,    75,    -1,    -1,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      54,    -1,   106,   107,    -1,    -1,    -1,     3,    -1,    -1,
      -1,    -1,    -1,    -1,    68,    -1,    -1,    -1,    -1,    73,
      74,    75,    -1,    -1,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      46,    -1,   106,   107,    -1,    -1,    -1,    -1,    54,    -1,
      -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,    74,    75,
      -1,    -1,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    46,    -1,
     106,   107,    -1,    -1,    -1,    -1,    54,    -1,    -1,    -1,
      -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    75,    -1,    -1,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    -1,    -1,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    46,    -1,   106,   107,
      -1,    -1,    -1,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    73,    74,    75,    -1,    -1,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    -1,    -1,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,    46,    -1,   106,   107,    -1,    -1,
      -1,     3,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    73,    74,    75,    -1,    -1,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,     3,
      -1,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    54,    -1,   106,   107,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    73,    74,    75,    -1,    -1,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,     3,
      54,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    -1,    -1,   106,   107,    -1,    -1,    -1,    73,
      74,    75,    -1,    -1,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,     3,    -1,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      54,    -1,   106,   107,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,
      74,    75,    -1,    -1,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,     3,    54,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      -1,    -1,   106,   107,    -1,    -1,    -1,    73,    74,    75,
      -1,    -1,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,     3,    -1,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    54,    -1,
     106,   107,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,    74,    75,
      -1,    -1,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,     3,    54,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    -1,    -1,
     106,   107,    -1,    -1,    -1,    73,    74,    75,    -1,    -1,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,     3,    -1,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    54,    -1,   106,   107,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    75,    -1,    -1,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,     3,    54,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    -1,    -1,   106,   107,
      -1,    -1,    -1,    73,    74,    75,    -1,    -1,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,     3,    -1,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,    54,    -1,   106,   107,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    73,    74,    75,    -1,    -1,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,     3,    54,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,    -1,    -1,   106,   107,    -1,    -1,
      -1,    73,    74,    75,    -1,    -1,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,     3,
      -1,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    54,    -1,   106,   107,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    73,    74,    75,    -1,    -1,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,     3,
      54,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    -1,    -1,   106,   107,    -1,    -1,    -1,    73,
      74,    75,    -1,    -1,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,     3,    -1,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      54,    -1,   106,   107,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,
      74,    75,    -1,    -1,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,     3,    54,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      -1,    -1,   106,   107,    -1,    -1,    -1,    73,    74,    75,
      -1,    -1,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,     3,    -1,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    54,    -1,
     106,   107,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,    74,    75,
      -1,    -1,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,     3,    54,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    -1,    -1,
     106,   107,    -1,    -1,    -1,    73,    74,    75,    -1,    -1,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,     3,    -1,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    54,    -1,   106,   107,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    73,    74,    75,    -1,    -1,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,     3,    54,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    -1,    -1,   106,   107,
      -1,    -1,    -1,    73,    74,    75,    -1,    -1,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,     3,    -1,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,    54,    -1,   106,   107,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    73,    74,    75,    -1,    -1,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,     3,    54,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,    -1,    -1,   106,   107,    -1,    -1,
      -1,    73,    74,    75,    -1,    -1,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,     3,
      -1,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    54,    -1,   106,   107,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    73,    74,    75,    -1,    -1,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,     3,
      54,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    -1,    -1,   106,   107,    -1,    -1,    -1,    73,
      74,    75,    -1,    -1,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      54,    -1,   106,   107,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,
      74,    75,    -1,    -1,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      -1,    -1,   106,   107,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    37,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    46,    -1,    -1,    -1,
      50,    26,    27,    -1,    54,    55,    56,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    69,
      -1,    46,    -1,    -1,    -1,    50,    76,    77,    -1,    54,
      -1,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    91,    92,    -1,    -1,    -1,    96,    -1,    -1,    -1,
      -1,    76,    77,    -1,   104,    -1,   106,   107,   108,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,
      -1,    96,    -1,    -1,    -1,   100,    -1,    -1,    -1,   104,
      -1,   106,   107,   108,     4,     5,     6,     7,     8,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    37,    -1,    -1,
      -1,    -1,    -1,    -1,    37,    -1,    46,    -1,    -1,    -1,
      50,    -1,    -1,    -1,    54,    -1,    56,    50,    -1,    -1,
      -1,    54,    55,    56,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    -1,    -1,
      -1,    -1,    -1,    76,    77,    -1,    -1,    -1,    -1,    -1,
      -1,    91,    92,    -1,    -1,    -1,    96,    -1,    91,    92,
     100,    -1,    -1,    96,   104,    -1,   106,   107,   108,    -1,
      -1,   104,    -1,   106,   107,   108,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    37,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    50,    26,    27,    -1,    54,    55,    56,    -1,
      -1,    -1,    26,    27,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    37,    47,    -1,    -1,    50,    76,    77,
      -1,    54,    -1,    56,    -1,    -1,    50,    -1,    -1,    53,
      54,    -1,    56,    91,    92,    -1,    -1,    -1,    96,    -1,
      -1,    -1,    -1,    76,    77,    -1,   104,    -1,   106,   107,
     108,    -1,    76,    77,    -1,    -1,    -1,    -1,    91,    92,
      -1,    -1,    -1,    96,    -1,    -1,    -1,    91,    92,    -1,
      -1,   104,    96,   106,   107,   108,    -1,    -1,    -1,    -1,
     104,    -1,   106,   107,   108,     4,     5,     6,     7,     8,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    37,    -1,
      -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,
      -1,    50,    -1,    -1,    -1,    54,    55,    56,    50,    -1,
      -1,    -1,    54,    -1,    56,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    -1,
      -1,    -1,    -1,    -1,    76,    77,    -1,    -1,    -1,    -1,
      -1,    -1,    91,    92,    -1,    -1,    -1,    96,    -1,    91,
      92,    -1,    -1,    -1,    96,   104,    -1,   106,   107,   108,
      -1,    -1,   104,    54,   106,   107,   108,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    86,    87,    88,    89,    90,
      37,    -1,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,    50,    -1,   106,   107,    54,    -1,    56,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    -1,    -1,    -1,    -1,    46,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    54,    91,    92,    -1,    -1,    -1,    96,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   104,    69,   106,
     107,   108,    73,    74,    75,    -1,    -1,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      -1,    -1,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,    46,    -1,   106,   107,    -1,    -1,    -1,
      -1,    54,    55,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    74,    75,    -1,    -1,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,    46,    -1,   106,   107,    -1,    -1,    -1,    -1,    54,
      55,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,    74,
      75,    -1,    -1,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    53,    54,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,    -1,
      -1,   106,   107,    -1,    -1,    -1,    73,    74,    75,    -1,
      -1,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    54,    55,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,    -1,    -1,   106,
     107,    -1,    -1,    73,    74,    75,    -1,    -1,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    54,    -1,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,    -1,    -1,   106,   107,    71,    -1,
      73,    74,    75,    -1,    -1,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    54,    55,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,    -1,    -1,   106,   107,    -1,    -1,    73,    74,    75,
      -1,    -1,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    54,    -1,
     106,   107,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    69,    -1,    -1,    -1,    73,    74,    75,
      -1,    -1,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    54,    55,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    -1,    -1,
     106,   107,    -1,    -1,    73,    74,    75,    -1,    -1,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    54,    55,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,    -1,    -1,   106,   107,    -1,
      -1,    73,    74,    75,    -1,    -1,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    54,
      -1,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    -1,    -1,   106,   107,    -1,    -1,    73,    74,
      75,    -1,    -1,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    54,    -1,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,    -1,
      -1,   106,   107,    -1,    -1,    -1,    74,    75,    -1,    -1,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    54,    -1,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    -1,    -1,   106,   107,
      -1,    -1,    -1,    -1,    75,    -1,    -1,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      54,    -1,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,    -1,    -1,   106,   107,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      -1,    -1,   106,   107
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   110,   111,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    45,    47,    49,
      50,    51,    52,    54,    56,    76,    77,    81,    91,    92,
      96,   102,   104,   106,   107,   108,   112,   113,   114,   115,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   134,   135,   136,   138,
     139,   147,   148,   149,   151,   152,   153,   158,   159,   160,
     167,   169,   182,   184,   193,   194,   196,   203,   204,   205,
     207,   209,   217,   218,   219,   220,   222,   225,   226,   227,
     230,   248,   253,   257,   258,   259,   260,   261,   262,   263,
     264,   269,   272,   273,   274,     3,     3,     1,   116,   259,
       1,   261,   262,     1,     3,     1,     3,    14,     1,   262,
       1,   259,   261,   277,     1,   262,     1,     1,   262,     1,
     262,   275,     1,     3,    46,     1,   262,     1,     6,     1,
       6,     1,     3,   262,   254,   270,     1,     6,     7,     1,
     262,   264,     1,     6,     1,     3,    46,     1,   262,     1,
       3,     6,   221,     1,     6,   223,   224,     1,     6,     1,
     262,    55,   262,   276,    46,   262,    46,    55,    69,   262,
     275,   278,   262,   261,     1,     3,   275,   262,   262,   262,
       1,     3,   275,   262,   262,   262,   262,   133,    32,    34,
      47,   115,   137,   115,   150,   115,   168,   183,   195,    48,
     212,   215,   216,   115,     1,    54,     1,     6,   228,   229,
     228,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    72,    85,   263,     3,    54,    68,    73,
      74,    75,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   106,   107,    56,    57,
     263,     3,     3,    72,    85,     3,    46,     3,    46,     3,
       3,     3,     3,    46,     3,    46,     3,    46,    85,    72,
      88,     3,     3,     3,     3,     3,     3,     1,    71,    72,
       3,   115,     3,     3,     3,   231,     3,   249,     3,     3,
       6,   255,   256,     1,     6,   210,   211,   271,     3,     3,
       3,     3,     3,     3,    85,     3,    46,     3,     3,    88,
       3,     3,    72,     3,    84,     3,    72,     3,     3,     3,
       1,    55,   262,    53,    55,   262,    55,    46,    69,     1,
      55,     1,    55,    72,    84,    85,     3,     3,     3,     3,
     146,   146,   146,   170,   185,   146,     1,     3,    46,   146,
     213,   214,     3,     1,   210,     3,     3,    72,     9,   228,
       3,   100,   262,     6,   262,   262,   262,   262,   262,   262,
     262,   262,   262,   262,   259,   277,    47,   108,   262,   266,
     275,    53,   275,   262,     1,   262,   262,   262,   262,   262,
     262,   262,   262,   262,   262,   262,     6,   262,   262,   262,
     262,   262,   262,   262,   262,   262,   262,   262,   262,   262,
     262,   262,   100,   262,     6,   259,   262,   262,   259,     1,
     262,     3,   262,   262,     1,    54,   232,   233,     1,    33,
     235,   250,     3,    72,     3,    72,    69,     1,   258,     1,
     262,     6,     4,     6,   224,    55,    55,   262,   262,    55,
     262,   262,   262,     9,   115,    16,    17,   140,   142,   143,
     145,     9,     1,     3,    23,    24,   171,   176,   178,     1,
       3,    23,   176,   186,    30,   197,   198,   199,   200,     3,
      46,     9,   146,   115,   208,     1,    53,     6,     3,     3,
     262,    55,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,    85,   267,   108,     3,     3,     3,     1,    53,
       3,     1,    46,   262,    55,    85,    71,     3,     3,    46,
       3,     3,   210,   241,   235,     3,     6,   236,   237,     3,
     251,     1,   256,   211,   262,     3,     3,     3,     3,    55,
      69,     3,     1,     3,     1,   262,     9,   141,   144,     3,
       3,     1,     4,     6,     7,     8,   180,   181,     1,     9,
     177,     3,     1,     4,     6,   191,   192,     9,     1,     3,
       4,     6,    88,   201,   202,     9,   199,   146,     3,     9,
      53,   206,     3,    46,    55,   262,   275,     1,    54,   268,
       3,   265,     1,   262,    55,   262,   262,   154,   155,     1,
      53,     3,     6,    38,    47,    48,    90,   204,   242,   243,
     245,   246,     3,    54,   238,    72,     3,   204,   243,   245,
     246,   252,   262,     3,     3,     3,     3,   146,   146,     3,
      46,    71,     3,    46,    72,     3,     3,    46,   179,     3,
      46,     3,    46,    72,     3,     3,   259,     3,    72,    88,
       3,     3,    46,    53,     3,     3,     3,   210,   212,    53,
       3,    46,    70,    19,    20,    21,   115,   156,   157,   161,
     163,   165,   115,   234,    85,     3,     6,     1,     6,    76,
     247,     9,     6,    27,   239,   240,   258,   237,     9,   140,
     174,   175,     4,   172,   173,   181,   146,   115,   189,   190,
     187,   188,   192,     3,   202,   259,     3,     1,    53,   146,
     262,     1,     3,    46,     1,     3,    46,     1,     3,    46,
       9,   156,    53,   262,   244,    85,     3,     6,     3,    72,
       3,    53,    72,     3,   146,   115,   146,   115,   146,   115,
     146,   115,     3,     3,     9,     3,    46,     3,   162,   115,
       3,   164,   115,     3,   166,   115,     3,     3,   212,   262,
       6,    76,   240,   146,   146,   146,   146,     3,     6,     9,
       9,     9,     9,     3,     3,     3,     3
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
#line 204 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); ;}
    break;

  case 7:
#line 205 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); ;}
    break;

  case 8:
#line 211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      ;}
    break;

  case 10:
#line 217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 11:
#line 222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 12:
#line 227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 13:
#line 232 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 17:
#line 243 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 18:
#line 249 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 19:
#line 255 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      ;}
    break;

  case 20:
#line 262 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); ;}
    break;

  case 21:
#line 263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; ;}
    break;

  case 22:
#line 266 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; ;}
    break;

  case 23:
#line 267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; ;}
    break;

  case 24:
#line 268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; ;}
    break;

  case 25:
#line 269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;;}
    break;

  case 26:
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   ;}
    break;

  case 27:
#line 279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 28:
#line 286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); ;}
    break;

  case 50:
#line 312 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; ;}
    break;

  case 51:
#line 314 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); ;}
    break;

  case 52:
#line 319 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 53:
#line 323 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtUnref( LINE, (yyvsp[(1) - (5)].fal_val) );
   ;}
    break;

  case 54:
#line 327 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) );
      ;}
    break;

  case 55:
#line 331 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 56:
#line 335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), (yyvsp[(5) - (6)].fal_val) );
      ;}
    break;

  case 57:
#line 340 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) ) );
      ;}
    break;

  case 69:
#line 364 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoAdd( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 70:
#line 371 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSub( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 71:
#line 378 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMul( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 72:
#line 385 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoDiv( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 73:
#line 392 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMod( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 74:
#line 399 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoPow( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 75:
#line 406 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBAND( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 76:
#line 413 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 77:
#line 420 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBXOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 78:
#line 426 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHL( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 79:
#line 432 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 80:
#line 440 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      ;}
    break;

  case 81:
#line 447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 82:
#line 454 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      ;}
    break;

  case 83:
#line 462 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 84:
#line 463 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 85:
#line 464 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; ;}
    break;

  case 86:
#line 468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 87:
#line 469 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 88:
#line 470 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 89:
#line 474 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      ;}
    break;

  case 90:
#line 482 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 91:
#line 489 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 92:
#line 499 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 93:
#line 500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; ;}
    break;

  case 94:
#line 504 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 95:
#line 505 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 98:
#line 512 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      ;}
    break;

  case 101:
#line 522 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); ;}
    break;

  case 102:
#line 527 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      ;}
    break;

  case 104:
#line 539 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 105:
#line 540 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; ;}
    break;

  case 107:
#line 545 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   ;}
    break;

  case 108:
#line 552 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 561 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 110:
#line 569 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 579 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 588 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 113:
#line 595 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>( (yyvsp[(1) - (1)].fal_stat) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 114:
#line 602 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 115:
#line 610 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 625 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 117:
#line 629 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 118:
#line 634 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, 0, 0, 0 );
      ;}
    break;

  case 119:
#line 641 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 120:
#line 645 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 121:
#line 650 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for, "", CURRENT_LINE );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, 0, 0, 0 );
      ;}
    break;

  case 122:
#line 659 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 124:
#line 683 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 699 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 709 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_forin ); ;}
    break;

  case 129:
#line 718 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 133:
#line 732 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 745 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 135:
#line 753 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 136:
#line 757 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 137:
#line 763 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 138:
#line 769 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      ;}
    break;

  case 139:
#line 776 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 140:
#line 781 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 141:
#line 790 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   ;}
    break;

  case 142:
#line 799 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 811 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 144:
#line 813 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 822 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 146:
#line 826 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 838 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 148:
#line 839 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 848 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 150:
#line 852 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->allBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->allBlock() );
      ;}
    break;

  case 151:
#line 864 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 152:
#line 866 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->allBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forall );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->allBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 153:
#line 875 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forall ); ;}
    break;

  case 154:
#line 879 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 155:
#line 887 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 156:
#line 896 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 157:
#line 898 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 160:
#line 907 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 162:
#line 913 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 164:
#line 923 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 165:
#line 931 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 166:
#line 935 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 947 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 957 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 171:
#line 966 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 980 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 177:
#line 984 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 180:
#line 996 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 181:
#line 1005 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1017 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1028 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1039 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1059 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 186:
#line 1067 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 187:
#line 1076 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 188:
#line 1078 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 191:
#line 1087 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 193:
#line 1093 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 195:
#line 1103 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 196:
#line 1112 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 197:
#line 1116 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1128 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1138 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 204:
#line 1152 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1164 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1185 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_val), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 207:
#line 1189 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 208:
#line 1193 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 209:
#line 1201 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 210:
#line 1208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 211:
#line 1218 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 213:
#line 1227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 219:
#line 1247 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 222:
#line 1295 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1306 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 226:
#line 1319 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1331 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1353 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 229:
#line 1354 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 230:
#line 1366 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 231:
#line 1372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 233:
#line 1381 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 234:
#line 1382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 235:
#line 1385 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 237:
#line 1390 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 238:
#line 1391 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 239:
#line 1398 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 246:
#line 1482 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 247:
#line 1487 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 248:
#line 1493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 250:
#line 1502 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 252:
#line 1507 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 253:
#line 1517 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 254:
#line 1520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 255:
#line 1529 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 256:
#line 1536 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1546 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 258:
#line 1552 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 259:
#line 1564 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1574 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 261:
#line 1579 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 262:
#line 1591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1600 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 264:
#line 1607 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 265:
#line 1615 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 266:
#line 1620 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 267:
#line 1632 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 268:
#line 1637 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     ;}
    break;

  case 271:
#line 1650 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      ;}
    break;

  case 272:
#line 1654 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      ;}
    break;

  case 273:
#line 1668 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 274:
#line 1675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 276:
#line 1683 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 278:
#line 1687 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 280:
#line 1693 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 281:
#line 1697 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 284:
#line 1706 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 285:
#line 1717 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1751 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1779 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 291:
#line 1787 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 292:
#line 1788 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 297:
#line 1805 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1838 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 299:
#line 1843 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
   ;}
    break;

  case 300:
#line 1849 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 301:
#line 1850 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 303:
#line 1856 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1864 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 308:
#line 1874 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 309:
#line 1877 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1899 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1923 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 313:
#line 1934 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1956 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1986 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 318:
#line 1993 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 319:
#line 2001 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 320:
#line 2007 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 321:
#line 2013 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 322:
#line 2026 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 2066 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 2091 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 329:
#line 2103 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 330:
#line 2106 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 2134 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 333:
#line 2139 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 2154 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 337:
#line 2161 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 338:
#line 2176 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 339:
#line 2177 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 340:
#line 2178 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 341:
#line 2188 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 342:
#line 2189 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 343:
#line 2190 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 344:
#line 2191 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 345:
#line 2196 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 347:
#line 2214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 348:
#line 2215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 350:
#line 2227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 351:
#line 2232 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 352:
#line 2237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 353:
#line 2243 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 356:
#line 2258 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 357:
#line 2259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 358:
#line 2260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 359:
#line 2261 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 360:
#line 2262 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 361:
#line 2263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 362:
#line 2264 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 363:
#line 2265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 364:
#line 2266 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 365:
#line 2267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 366:
#line 2268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 367:
#line 2269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 368:
#line 2270 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 369:
#line 2271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 370:
#line 2273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 371:
#line 2275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 372:
#line 2276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 373:
#line 2277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 374:
#line 2278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 375:
#line 2279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 376:
#line 2280 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 377:
#line 2281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 378:
#line 2282 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 379:
#line 2283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 380:
#line 2284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 381:
#line 2285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 382:
#line 2286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 383:
#line 2287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 384:
#line 2288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 385:
#line 2289 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 386:
#line 2290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 387:
#line 2291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 388:
#line 2292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 389:
#line 2293 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 390:
#line 2294 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 391:
#line 2295 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 392:
#line 2296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 393:
#line 2297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 396:
#line 2300 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 397:
#line 2308 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 398:
#line 2312 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 399:
#line 2316 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 404:
#line 2324 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 405:
#line 2329 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 406:
#line 2332 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 407:
#line 2335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 408:
#line 2338 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 409:
#line 2345 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      ;}
    break;

  case 410:
#line 2351 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      ;}
    break;

  case 411:
#line 2355 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 412:
#line 2356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 413:
#line 2365 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 414:
#line 2398 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         ;}
    break;

  case 416:
#line 2409 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      ;}
    break;

  case 417:
#line 2413 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 418:
#line 2421 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 419:
#line 2452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->checkLocalUndefined();
            COMPILER->closeFunction();
         ;}
    break;

  case 421:
#line 2466 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      ;}
    break;

  case 422:
#line 2475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 423:
#line 2480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 424:
#line 2487 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 425:
#line 2494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 426:
#line 2503 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 427:
#line 2505 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 428:
#line 2509 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 429:
#line 2515 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 430:
#line 2517 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 431:
#line 2521 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 432:
#line 2529 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 433:
#line 2530 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); ;}
    break;

  case 434:
#line 2532 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      ;}
    break;

  case 435:
#line 2539 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 436:
#line 2540 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 437:
#line 2544 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 438:
#line 2545 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (2)].fal_adecl)->pushBack( (yyvsp[(2) - (2)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (2)].fal_adecl); ;}
    break;

  case 439:
#line 2549 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 440:
#line 2556 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 441:
#line 2563 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 442:
#line 2564 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6279 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2568 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


