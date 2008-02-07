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
     IMPORT = 300,
     DIRECTIVE = 301,
     COLON = 302,
     FUNCDECL = 303,
     STATIC = 304,
     FORDOT = 305,
     LISTPAR = 306,
     LOOP = 307,
     TRUE_TOKEN = 308,
     FALSE_TOKEN = 309,
     OUTER_STRING = 310,
     CLOSEPAR = 311,
     OPENPAR = 312,
     CLOSESQUARE = 313,
     OPENSQUARE = 314,
     DOT = 315,
     ASSIGN_POW = 316,
     ASSIGN_SHL = 317,
     ASSIGN_SHR = 318,
     ASSIGN_BXOR = 319,
     ASSIGN_BOR = 320,
     ASSIGN_BAND = 321,
     ASSIGN_MOD = 322,
     ASSIGN_DIV = 323,
     ASSIGN_MUL = 324,
     ASSIGN_SUB = 325,
     ASSIGN_ADD = 326,
     ARROW = 327,
     FOR_STEP = 328,
     OP_TO = 329,
     COMMA = 330,
     QUESTION = 331,
     OR = 332,
     AND = 333,
     NOT = 334,
     LET = 335,
     LE = 336,
     GE = 337,
     LT = 338,
     GT = 339,
     NEQ = 340,
     EEQ = 341,
     OP_EQ = 342,
     OP_ASSIGN = 343,
     PROVIDES = 344,
     OP_NOTIN = 345,
     OP_IN = 346,
     HASNT = 347,
     HAS = 348,
     DIESIS = 349,
     ATSIGN = 350,
     CAP = 351,
     VBAR = 352,
     AMPER = 353,
     MINUS = 354,
     PLUS = 355,
     PERCENT = 356,
     SLASH = 357,
     STAR = 358,
     POW = 359,
     SHR = 360,
     SHL = 361,
     BANG = 362,
     NEG = 363,
     DECREMENT = 364,
     INCREMENT = 365,
     DOLLAR = 366
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
#define IMPORT 300
#define DIRECTIVE 301
#define COLON 302
#define FUNCDECL 303
#define STATIC 304
#define FORDOT 305
#define LISTPAR 306
#define LOOP 307
#define TRUE_TOKEN 308
#define FALSE_TOKEN 309
#define OUTER_STRING 310
#define CLOSEPAR 311
#define OPENPAR 312
#define CLOSESQUARE 313
#define OPENSQUARE 314
#define DOT 315
#define ASSIGN_POW 316
#define ASSIGN_SHL 317
#define ASSIGN_SHR 318
#define ASSIGN_BXOR 319
#define ASSIGN_BOR 320
#define ASSIGN_BAND 321
#define ASSIGN_MOD 322
#define ASSIGN_DIV 323
#define ASSIGN_MUL 324
#define ASSIGN_SUB 325
#define ASSIGN_ADD 326
#define ARROW 327
#define FOR_STEP 328
#define OP_TO 329
#define COMMA 330
#define QUESTION 331
#define OR 332
#define AND 333
#define NOT 334
#define LET 335
#define LE 336
#define GE 337
#define LT 338
#define GT 339
#define NEQ 340
#define EEQ 341
#define OP_EQ 342
#define OP_ASSIGN 343
#define PROVIDES 344
#define OP_NOTIN 345
#define OP_IN 346
#define HASNT 347
#define HAS 348
#define DIESIS 349
#define ATSIGN 350
#define CAP 351
#define VBAR 352
#define AMPER 353
#define MINUS 354
#define PLUS 355
#define PERCENT 356
#define SLASH 357
#define STAR 358
#define POW 359
#define SHR 360
#define SHL 361
#define BANG 362
#define NEG 363
#define DECREMENT 364
#define INCREMENT 365
#define DOLLAR 366




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
#line 382 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 395 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   6572

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  112
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  173
/* YYNRULES -- Number of rules.  */
#define YYNRULES  451
/* YYNRULES -- Number of states.  */
#define YYNSTATES  831

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   366

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
     105,   106,   107,   108,   109,   110,   111
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    42,
      46,    50,    52,    54,    58,    62,    66,    69,    73,    79,
      82,    84,    86,    88,    90,    92,    94,    96,    98,   100,
     102,   104,   106,   108,   110,   112,   114,   116,   118,   120,
     122,   124,   128,   132,   137,   143,   148,   153,   160,   167,
     169,   171,   173,   175,   177,   179,   181,   183,   185,   187,
     189,   194,   199,   204,   209,   214,   219,   224,   229,   234,
     239,   244,   245,   251,   254,   258,   260,   264,   268,   271,
     275,   276,   283,   286,   290,   294,   298,   302,   303,   305,
     306,   310,   313,   317,   318,   323,   327,   331,   332,   335,
     338,   342,   345,   349,   353,   354,   360,   363,   371,   381,
     385,   393,   403,   407,   408,   418,   419,   427,   433,   434,
     437,   439,   441,   443,   445,   449,   453,   457,   460,   464,
     467,   471,   475,   477,   478,   485,   489,   493,   494,   501,
     505,   509,   510,   517,   521,   525,   526,   533,   537,   541,
     542,   545,   549,   551,   552,   558,   559,   565,   566,   572,
     573,   579,   580,   581,   585,   586,   588,   591,   594,   597,
     599,   603,   605,   607,   609,   613,   615,   616,   623,   627,
     631,   632,   635,   639,   641,   642,   648,   649,   655,   656,
     662,   663,   669,   671,   675,   676,   678,   680,   686,   691,
     695,   699,   700,   707,   710,   714,   715,   717,   719,   722,
     725,   728,   733,   737,   743,   747,   749,   753,   755,   757,
     761,   765,   771,   774,   780,   781,   789,   793,   799,   800,
     807,   810,   811,   813,   817,   819,   820,   821,   827,   828,
     832,   835,   839,   842,   846,   850,   854,   858,   864,   870,
     874,   880,   886,   890,   893,   897,   901,   903,   907,   911,
     915,   917,   921,   925,   929,   931,   935,   939,   943,   948,
     952,   955,   959,   962,   966,   967,   969,   973,   976,   980,
     983,   984,   993,   997,  1000,  1001,  1005,  1006,  1012,  1013,
    1016,  1018,  1022,  1025,  1026,  1030,  1032,  1036,  1038,  1040,
    1042,  1043,  1046,  1048,  1050,  1052,  1054,  1055,  1063,  1069,
    1074,  1075,  1079,  1083,  1085,  1088,  1092,  1097,  1098,  1107,
    1110,  1113,  1114,  1117,  1119,  1121,  1123,  1125,  1126,  1131,
    1133,  1137,  1141,  1143,  1146,  1150,  1154,  1156,  1158,  1160,
    1162,  1164,  1166,  1168,  1170,  1172,  1174,  1176,  1179,  1184,
    1190,  1194,  1196,  1198,  1202,  1205,  1209,  1213,  1217,  1221,
    1225,  1229,  1233,  1237,  1241,  1245,  1248,  1253,  1258,  1262,
    1265,  1268,  1271,  1274,  1278,  1282,  1286,  1290,  1294,  1298,
    1302,  1306,  1310,  1313,  1317,  1321,  1325,  1329,  1333,  1336,
    1339,  1342,  1344,  1346,  1350,  1355,  1361,  1364,  1366,  1368,
    1370,  1372,  1376,  1380,  1385,  1390,  1396,  1401,  1405,  1406,
    1413,  1414,  1421,  1426,  1430,  1433,  1434,  1440,  1442,  1445,
    1451,  1457,  1462,  1466,  1469,  1473,  1477,  1480,  1484,  1488,
    1492,  1496,  1501,  1503,  1507,  1509,  1512,  1514,  1518,  1520,
    1524,  1528
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     113,     0,    -1,   114,    -1,    -1,   114,   115,    -1,   116,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   117,    -1,
     227,    -1,   207,    -1,   235,    -1,   253,    -1,   118,    -1,
     222,    -1,   223,    -1,   225,    -1,   230,    -1,    39,     6,
       3,    -1,    39,     7,     3,    -1,    39,     1,     3,    -1,
     120,    -1,     3,    -1,    48,     1,     3,    -1,    34,     1,
       3,    -1,    32,     1,     3,    -1,     1,     3,    -1,   264,
      88,   267,    -1,   119,    75,   264,    88,   267,    -1,   267,
       3,    -1,   121,    -1,   122,    -1,   123,    -1,   135,    -1,
     152,    -1,   156,    -1,   170,    -1,   185,    -1,   139,    -1,
     150,    -1,   151,    -1,   196,    -1,   197,    -1,   206,    -1,
     262,    -1,   258,    -1,   220,    -1,   221,    -1,   161,    -1,
     162,    -1,   163,    -1,    10,   119,     3,    -1,    10,     1,
       3,    -1,   266,    88,   267,     3,    -1,   266,    88,   111,
     111,     3,    -1,   266,    88,   280,     3,    -1,   266,    88,
     271,     3,    -1,   266,    75,   283,    88,   267,     3,    -1,
     266,    75,   283,    88,   280,     3,    -1,   124,    -1,   125,
      -1,   126,    -1,   127,    -1,   128,    -1,   129,    -1,   130,
      -1,   131,    -1,   132,    -1,   133,    -1,   134,    -1,   267,
      71,   267,     3,    -1,   266,    70,   267,     3,    -1,   266,
      69,   267,     3,    -1,   266,    68,   267,     3,    -1,   266,
      67,   267,     3,    -1,   266,    61,   267,     3,    -1,   266,
      66,   267,     3,    -1,   266,    65,   267,     3,    -1,   266,
      64,   267,     3,    -1,   266,    62,   267,     3,    -1,   266,
      63,   267,     3,    -1,    -1,   137,   136,   149,     9,     3,
      -1,   138,   118,    -1,    11,   267,     3,    -1,    52,    -1,
      11,     1,     3,    -1,    11,   267,    47,    -1,    52,    47,
      -1,    11,     1,    47,    -1,    -1,   141,   140,   149,   143,
       9,     3,    -1,   142,   118,    -1,    15,   267,     3,    -1,
      15,     1,     3,    -1,    15,   267,    47,    -1,    15,     1,
      47,    -1,    -1,   146,    -1,    -1,   145,   144,   149,    -1,
      16,     3,    -1,    16,     1,     3,    -1,    -1,   148,   147,
     149,   143,    -1,    17,   267,     3,    -1,    17,     1,     3,
      -1,    -1,   149,   118,    -1,    12,     3,    -1,    12,     1,
       3,    -1,    13,     3,    -1,    13,    14,     3,    -1,    13,
       1,     3,    -1,    -1,   154,   153,   149,     9,     3,    -1,
     155,   118,    -1,    18,   266,    88,   267,    74,   267,     3,
      -1,    18,   266,    88,   267,    74,   267,    73,   267,     3,
      -1,    18,     1,     3,    -1,    18,   266,    88,   267,    74,
     267,    47,    -1,    18,   266,    88,   267,    74,   267,    73,
     267,    47,    -1,    18,     1,    47,    -1,    -1,    18,   282,
      91,   267,     3,   157,   159,     9,     3,    -1,    -1,    18,
     282,    91,   267,    47,   158,   118,    -1,    18,   282,    91,
       1,     3,    -1,    -1,   160,   159,    -1,   118,    -1,   164,
      -1,   166,    -1,   168,    -1,    50,   267,     3,    -1,    50,
       1,     3,    -1,   105,   280,     3,    -1,   105,     3,    -1,
      84,   280,     3,    -1,    84,     3,    -1,   105,     1,     3,
      -1,    84,     1,     3,    -1,    55,    -1,    -1,    19,     3,
     165,   149,     9,     3,    -1,    19,    47,   118,    -1,    19,
       1,     3,    -1,    -1,    20,     3,   167,   149,     9,     3,
      -1,    20,    47,   118,    -1,    20,     1,     3,    -1,    -1,
      21,     3,   169,   149,     9,     3,    -1,    21,    47,   118,
      -1,    21,     1,     3,    -1,    -1,   172,   171,   173,   179,
       9,     3,    -1,    22,   267,     3,    -1,    22,     1,     3,
      -1,    -1,   173,   174,    -1,   173,     1,     3,    -1,     3,
      -1,    -1,    23,   183,     3,   175,   149,    -1,    -1,    23,
     183,    47,   176,   118,    -1,    -1,    23,     1,     3,   177,
     149,    -1,    -1,    23,     1,    47,   178,   118,    -1,    -1,
      -1,   181,   180,   182,    -1,    -1,    24,    -1,    24,     1,
      -1,     3,   149,    -1,    47,   118,    -1,   184,    -1,   183,
      75,   184,    -1,     8,    -1,     4,    -1,     7,    -1,     4,
      74,     4,    -1,     6,    -1,    -1,   187,   186,   188,   179,
       9,     3,    -1,    25,   267,     3,    -1,    25,     1,     3,
      -1,    -1,   188,   189,    -1,   188,     1,     3,    -1,     3,
      -1,    -1,    23,   194,     3,   190,   149,    -1,    -1,    23,
     194,    47,   191,   118,    -1,    -1,    23,     1,     3,   192,
     149,    -1,    -1,    23,     1,    47,   193,   118,    -1,   195,
      -1,   194,    75,   195,    -1,    -1,     4,    -1,     6,    -1,
      28,   280,    74,   267,     3,    -1,    28,   280,     1,     3,
      -1,    28,     1,     3,    -1,    29,    47,   118,    -1,    -1,
     199,   198,   149,   200,     9,     3,    -1,    29,     3,    -1,
      29,     1,     3,    -1,    -1,   201,    -1,   202,    -1,   201,
     202,    -1,   203,   149,    -1,    30,     3,    -1,    30,    91,
     264,     3,    -1,    30,   204,     3,    -1,    30,   204,    91,
     264,     3,    -1,    30,     1,     3,    -1,   205,    -1,   204,
      75,   205,    -1,     4,    -1,     6,    -1,    31,   267,     3,
      -1,    31,     1,     3,    -1,   208,   215,   149,     9,     3,
      -1,   210,   118,    -1,   212,    57,   213,    56,     3,    -1,
      -1,   212,    57,   213,     1,   209,    56,     3,    -1,   212,
       1,     3,    -1,   212,    57,   213,    56,    47,    -1,    -1,
     212,    57,     1,   211,    56,    47,    -1,    48,     6,    -1,
      -1,   214,    -1,   213,    75,   214,    -1,     6,    -1,    -1,
      -1,   218,   216,   149,     9,     3,    -1,    -1,   219,   217,
     118,    -1,    49,     3,    -1,    49,     1,     3,    -1,    49,
      47,    -1,    49,     1,    47,    -1,    40,   269,     3,    -1,
      40,     1,     3,    -1,    43,   267,     3,    -1,    43,   267,
      91,   267,     3,    -1,    43,   267,    91,     1,     3,    -1,
      43,     1,     3,    -1,    41,     6,    88,   263,     3,    -1,
      41,     6,    88,     1,     3,    -1,    41,     1,     3,    -1,
      44,     3,    -1,    44,   224,     3,    -1,    44,     1,     3,
      -1,     6,    -1,   224,    75,     6,    -1,    45,   226,     3,
      -1,    45,     1,     3,    -1,     6,    -1,   224,    75,     6,
      -1,    46,   228,     3,    -1,    46,     1,     3,    -1,   229,
      -1,   228,    75,   229,    -1,     6,    87,     6,    -1,     6,
      87,     4,    -1,   231,   234,     9,     3,    -1,   232,   233,
       3,    -1,    42,     3,    -1,    42,     1,     3,    -1,    42,
      47,    -1,    42,     1,    47,    -1,    -1,     6,    -1,   233,
      75,     6,    -1,   233,     3,    -1,   234,   233,     3,    -1,
       1,     3,    -1,    -1,    32,     6,   236,   237,   246,   251,
       9,     3,    -1,   238,   240,     3,    -1,     1,     3,    -1,
      -1,    57,   213,    56,    -1,    -1,    57,   213,     1,   239,
      56,    -1,    -1,    33,   241,    -1,   242,    -1,   241,    75,
     242,    -1,     6,   243,    -1,    -1,    57,   244,    56,    -1,
     245,    -1,   244,    75,   245,    -1,   263,    -1,     6,    -1,
      27,    -1,    -1,   246,   247,    -1,     3,    -1,   207,    -1,
     250,    -1,   248,    -1,    -1,    38,     3,   249,   215,   149,
       9,     3,    -1,    49,     6,    88,   267,     3,    -1,     6,
      88,   267,     3,    -1,    -1,    93,   252,     3,    -1,    93,
       1,     3,    -1,     6,    -1,    79,     6,    -1,   252,    75,
       6,    -1,   252,    75,    79,     6,    -1,    -1,    34,     6,
     254,   255,   256,   251,     9,     3,    -1,   240,     3,    -1,
       1,     3,    -1,    -1,   256,   257,    -1,     3,    -1,   207,
      -1,   250,    -1,   248,    -1,    -1,    36,   259,   260,     3,
      -1,   261,    -1,   260,    75,   261,    -1,   260,    75,     1,
      -1,     6,    -1,    35,     3,    -1,    35,   267,     3,    -1,
      35,     1,     3,    -1,     8,    -1,    53,    -1,    54,    -1,
       4,    -1,     5,    -1,     7,    -1,     6,    -1,   264,    -1,
      27,    -1,    26,    -1,   265,    -1,   266,   268,    -1,   266,
      59,   267,    58,    -1,   266,    59,   103,   267,    58,    -1,
     266,    60,     6,    -1,   263,    -1,   266,    -1,   267,   100,
     267,    -1,    99,   267,    -1,   267,    99,   267,    -1,   267,
     103,   267,    -1,   267,   102,   267,    -1,   267,   101,   267,
      -1,   267,   104,   267,    -1,   267,    98,   267,    -1,   267,
      97,   267,    -1,   267,    96,   267,    -1,   267,   106,   267,
      -1,   267,   105,   267,    -1,   107,   267,    -1,    80,   266,
      87,   267,    -1,    80,   266,    88,   267,    -1,   267,    85,
     267,    -1,   267,   110,    -1,   110,   267,    -1,   267,   109,
      -1,   109,   267,    -1,   267,    86,   267,    -1,   267,    87,
     267,    -1,   267,    88,   267,    -1,   267,    84,   267,    -1,
     267,    83,   267,    -1,   267,    82,   267,    -1,   267,    81,
     267,    -1,   267,    78,   267,    -1,   267,    77,   267,    -1,
      79,   267,    -1,   267,    93,   267,    -1,   267,    92,   267,
      -1,   267,    91,   267,    -1,   267,    90,   267,    -1,   267,
      89,     6,    -1,   111,   267,    -1,    95,   267,    -1,    94,
     267,    -1,   274,    -1,   269,    -1,   269,    60,     6,    -1,
     269,    59,   267,    58,    -1,   269,    59,   103,   267,    58,
      -1,   269,   268,    -1,   277,    -1,   278,    -1,   279,    -1,
     268,    -1,    57,   267,    56,    -1,    59,    47,    58,    -1,
      59,   267,    47,    58,    -1,    59,    47,   267,    58,    -1,
      59,   267,    47,   267,    58,    -1,   267,    57,   280,    56,
      -1,   267,    57,    56,    -1,    -1,   267,    57,   280,     1,
     270,    56,    -1,    -1,    48,   272,   273,   215,   149,     9,
      -1,    57,   213,    56,     3,    -1,    57,   213,     1,    -1,
       1,     3,    -1,    -1,    37,   275,   276,    72,   267,    -1,
     213,    -1,     1,     3,    -1,   267,    76,   267,    47,   267,
      -1,   267,    76,   267,    47,     1,    -1,   267,    76,   267,
       1,    -1,   267,    76,     1,    -1,    59,    58,    -1,    59,
     280,    58,    -1,    59,   280,     1,    -1,    51,    58,    -1,
      51,   281,    58,    -1,    51,   281,     1,    -1,    59,    72,
      58,    -1,    59,   284,    58,    -1,    59,   284,     1,    58,
      -1,   267,    -1,   280,    75,   267,    -1,   267,    -1,   281,
     267,    -1,   264,    -1,   282,    75,   264,    -1,   266,    -1,
     283,    75,   266,    -1,   267,    72,   267,    -1,   284,    75,
     267,    72,   267,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   198,   198,   201,   203,   207,   208,   209,   214,   219,
     220,   225,   230,   235,   240,   241,   242,   243,   247,   253,
     259,   267,   268,   271,   272,   273,   274,   279,   284,   291,
     292,   293,   294,   295,   296,   297,   298,   299,   300,   301,
     302,   303,   304,   305,   306,   307,   308,   309,   310,   311,
     312,   316,   318,   324,   328,   332,   336,   340,   345,   355,
     356,   357,   358,   359,   360,   361,   362,   363,   364,   365,
     369,   376,   383,   390,   397,   404,   411,   418,   425,   431,
     437,   445,   445,   459,   467,   468,   469,   473,   474,   475,
     479,   479,   494,   504,   505,   509,   510,   514,   516,   517,
     517,   526,   527,   532,   532,   544,   545,   548,   550,   556,
     565,   573,   583,   592,   600,   600,   614,   630,   634,   638,
     646,   650,   654,   664,   663,   688,   687,   713,   717,   719,
     723,   730,   731,   732,   736,   749,   757,   761,   767,   773,
     780,   785,   794,   804,   804,   818,   827,   831,   831,   844,
     853,   857,   857,   873,   882,   886,   886,   903,   904,   911,
     913,   914,   918,   920,   919,   930,   930,   942,   942,   954,
     954,   970,   973,   972,   985,   986,   987,   990,   991,   997,
     998,  1002,  1011,  1023,  1034,  1045,  1066,  1066,  1083,  1084,
    1091,  1093,  1094,  1098,  1100,  1099,  1110,  1110,  1123,  1123,
    1135,  1135,  1153,  1154,  1157,  1158,  1170,  1191,  1195,  1200,
    1208,  1215,  1214,  1233,  1234,  1237,  1239,  1243,  1244,  1248,
    1253,  1271,  1291,  1301,  1312,  1320,  1321,  1325,  1337,  1360,
    1361,  1368,  1378,  1387,  1388,  1388,  1392,  1396,  1397,  1397,
    1404,  1458,  1460,  1461,  1465,  1480,  1483,  1482,  1494,  1493,
    1508,  1509,  1513,  1514,  1523,  1527,  1535,  1542,  1552,  1558,
    1570,  1580,  1585,  1597,  1606,  1613,  1621,  1626,  1634,  1638,
    1646,  1651,  1663,  1668,  1676,  1677,  1681,  1685,  1697,  1704,
    1714,  1715,  1718,  1719,  1722,  1724,  1728,  1735,  1736,  1737,
    1749,  1748,  1807,  1810,  1816,  1818,  1819,  1819,  1825,  1827,
    1831,  1832,  1836,  1870,  1872,  1881,  1882,  1886,  1887,  1896,
    1899,  1901,  1905,  1906,  1909,  1927,  1931,  1931,  1965,  1987,
    2014,  2016,  2017,  2024,  2032,  2038,  2044,  2058,  2057,  2121,
    2122,  2128,  2130,  2134,  2135,  2138,  2157,  2166,  2165,  2183,
    2184,  2185,  2192,  2208,  2209,  2210,  2220,  2221,  2222,  2223,
    2224,  2225,  2229,  2247,  2248,  2249,  2260,  2261,  2266,  2271,
    2277,  2290,  2291,  2292,  2293,  2294,  2295,  2296,  2297,  2298,
    2299,  2300,  2301,  2302,  2303,  2304,  2305,  2307,  2309,  2310,
    2311,  2312,  2313,  2314,  2315,  2316,  2317,  2318,  2319,  2320,
    2321,  2322,  2323,  2324,  2325,  2326,  2327,  2328,  2329,  2330,
    2331,  2332,  2333,  2334,  2342,  2346,  2350,  2354,  2355,  2356,
    2357,  2358,  2363,  2366,  2369,  2372,  2378,  2384,  2389,  2389,
    2399,  2398,  2441,  2442,  2446,  2455,  2454,  2498,  2499,  2508,
    2513,  2520,  2527,  2537,  2538,  2542,  2549,  2550,  2554,  2563,
    2564,  2565,  2573,  2574,  2578,  2579,  2583,  2589,  2596,  2602,
    2609,  2610
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
  "EXPORT", "IMPORT", "DIRECTIVE", "COLON", "FUNCDECL", "STATIC", "FORDOT",
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
  "import_statement", "import_symbol_list", "directive_statement",
  "directive_pair_list", "directive_pair", "attributes_statement",
  "attributes_decl", "attributes_short_decl", "attribute_list",
  "attribute_vert_list", "class_decl", "@27", "class_def_inner",
  "class_param_list", "@28", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "inherit_param_list",
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
     365,   366
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   112,   113,   114,   114,   115,   115,   115,   116,   116,
     116,   116,   116,   116,   116,   116,   116,   116,   117,   117,
     117,   118,   118,   118,   118,   118,   118,   119,   119,   120,
     120,   120,   120,   120,   120,   120,   120,   120,   120,   120,
     120,   120,   120,   120,   120,   120,   120,   120,   120,   120,
     120,   121,   121,   122,   122,   122,   122,   122,   122,   123,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   136,   135,   135,   137,   137,   137,   138,   138,   138,
     140,   139,   139,   141,   141,   142,   142,   143,   143,   144,
     143,   145,   145,   147,   146,   148,   148,   149,   149,   150,
     150,   151,   151,   151,   153,   152,   152,   154,   154,   154,
     155,   155,   155,   157,   156,   158,   156,   156,   159,   159,
     160,   160,   160,   160,   161,   161,   162,   162,   162,   162,
     162,   162,   163,   165,   164,   164,   164,   167,   166,   166,
     166,   169,   168,   168,   168,   171,   170,   172,   172,   173,
     173,   173,   174,   175,   174,   176,   174,   177,   174,   178,
     174,   179,   180,   179,   181,   181,   181,   182,   182,   183,
     183,   184,   184,   184,   184,   184,   186,   185,   187,   187,
     188,   188,   188,   189,   190,   189,   191,   189,   192,   189,
     193,   189,   194,   194,   195,   195,   195,   196,   196,   196,
     197,   198,   197,   199,   199,   200,   200,   201,   201,   202,
     203,   203,   203,   203,   203,   204,   204,   205,   205,   206,
     206,   207,   207,   208,   209,   208,   208,   210,   211,   210,
     212,   213,   213,   213,   214,   215,   216,   215,   217,   215,
     218,   218,   219,   219,   220,   220,   221,   221,   221,   221,
     222,   222,   222,   223,   223,   223,   224,   224,   225,   225,
     226,   226,   227,   227,   228,   228,   229,   229,   230,   230,
     231,   231,   232,   232,   233,   233,   233,   234,   234,   234,
     236,   235,   237,   237,   238,   238,   239,   238,   240,   240,
     241,   241,   242,   243,   243,   244,   244,   245,   245,   245,
     246,   246,   247,   247,   247,   247,   249,   248,   250,   250,
     251,   251,   251,   252,   252,   252,   252,   254,   253,   255,
     255,   256,   256,   257,   257,   257,   257,   259,   258,   260,
     260,   260,   261,   262,   262,   262,   263,   263,   263,   263,
     263,   263,   264,   265,   265,   265,   266,   266,   266,   266,
     266,   267,   267,   267,   267,   267,   267,   267,   267,   267,
     267,   267,   267,   267,   267,   267,   267,   267,   267,   267,
     267,   267,   267,   267,   267,   267,   267,   267,   267,   267,
     267,   267,   267,   267,   267,   267,   267,   267,   267,   267,
     267,   267,   267,   267,   267,   267,   267,   267,   267,   267,
     267,   267,   268,   268,   268,   268,   269,   269,   270,   269,
     272,   271,   273,   273,   273,   275,   274,   276,   276,   277,
     277,   277,   277,   278,   278,   278,   278,   278,   278,   279,
     279,   279,   280,   280,   281,   281,   282,   282,   283,   283,
     284,   284
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     3,     3,
       3,     1,     1,     3,     3,     3,     2,     3,     5,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     3,     4,     5,     4,     4,     6,     6,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     0,     5,     2,     3,     1,     3,     3,     2,     3,
       0,     6,     2,     3,     3,     3,     3,     0,     1,     0,
       3,     2,     3,     0,     4,     3,     3,     0,     2,     2,
       3,     2,     3,     3,     0,     5,     2,     7,     9,     3,
       7,     9,     3,     0,     9,     0,     7,     5,     0,     2,
       1,     1,     1,     1,     3,     3,     3,     2,     3,     2,
       3,     3,     1,     0,     6,     3,     3,     0,     6,     3,
       3,     0,     6,     3,     3,     0,     6,     3,     3,     0,
       2,     3,     1,     0,     5,     0,     5,     0,     5,     0,
       5,     0,     0,     3,     0,     1,     2,     2,     2,     1,
       3,     1,     1,     1,     3,     1,     0,     6,     3,     3,
       0,     2,     3,     1,     0,     5,     0,     5,     0,     5,
       0,     5,     1,     3,     0,     1,     1,     5,     4,     3,
       3,     0,     6,     2,     3,     0,     1,     1,     2,     2,
       2,     4,     3,     5,     3,     1,     3,     1,     1,     3,
       3,     5,     2,     5,     0,     7,     3,     5,     0,     6,
       2,     0,     1,     3,     1,     0,     0,     5,     0,     3,
       2,     3,     2,     3,     3,     3,     3,     5,     5,     3,
       5,     5,     3,     2,     3,     3,     1,     3,     3,     3,
       1,     3,     3,     3,     1,     3,     3,     3,     4,     3,
       2,     3,     2,     3,     0,     1,     3,     2,     3,     2,
       0,     8,     3,     2,     0,     3,     0,     5,     0,     2,
       1,     3,     2,     0,     3,     1,     3,     1,     1,     1,
       0,     2,     1,     1,     1,     1,     0,     7,     5,     4,
       0,     3,     3,     1,     2,     3,     4,     0,     8,     2,
       2,     0,     2,     1,     1,     1,     1,     0,     4,     1,
       3,     3,     1,     2,     3,     3,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     2,     4,     5,
       3,     1,     1,     3,     2,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     2,     4,     4,     3,     2,
       2,     2,     2,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     2,     3,     3,     3,     3,     3,     2,     2,
       2,     1,     1,     3,     4,     5,     2,     1,     1,     1,
       1,     3,     3,     4,     4,     5,     4,     3,     0,     6,
       0,     6,     4,     3,     2,     0,     5,     1,     2,     5,
       5,     4,     3,     2,     3,     3,     2,     3,     3,     3,
       3,     4,     1,     3,     1,     2,     1,     3,     1,     3,
       3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    22,   349,   350,   352,   351,
     346,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   355,   354,     0,     0,     0,     0,     0,     0,   337,
     425,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    85,   347,   348,   142,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     4,
       5,     8,    13,    21,    30,    31,    32,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    33,    81,
       0,    38,    90,     0,    39,    40,    34,   114,     0,    35,
      48,    49,    50,    36,   155,    37,   186,    41,    42,   211,
      43,    10,   245,     0,     0,    46,    47,    14,    15,    16,
       9,    17,     0,   284,    11,    12,    45,    44,   361,   353,
     356,   362,     0,   410,   402,   401,   407,   408,   409,    26,
       6,     0,     0,     0,     0,   362,     0,     0,   109,     0,
     111,     0,     0,     0,     0,   353,     0,     0,     0,     0,
       0,     0,     0,     0,   442,     0,     0,   213,     0,     0,
       0,     0,   290,     0,   327,     0,   343,     0,     0,     0,
       0,     0,     0,     0,     0,   402,     0,     0,     0,   280,
     282,     0,     0,     0,   263,   266,     0,     0,   266,     0,
       0,     0,     0,     0,   274,     0,   240,     0,     0,   436,
     444,     0,    88,     0,     0,   433,     0,   442,     0,     0,
     392,     0,     0,   139,     0,   400,   399,   364,     0,   137,
       0,   375,   382,   380,   398,   107,     0,     0,     0,    83,
     107,    92,   107,   116,   159,   190,   107,     0,   107,   246,
     248,   232,     0,     0,     0,   285,     0,   284,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   357,    29,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   381,   379,     0,     0,   406,    52,
      51,     0,     0,    86,    89,    84,    87,   110,   113,   112,
      94,    96,    93,    95,   119,   122,     0,     0,     0,   158,
     157,     7,   189,   188,   209,     0,     0,     0,   214,   210,
     230,   229,    25,     0,    24,     0,   345,   344,   342,     0,
     339,     0,   244,   427,   242,     0,    20,    18,    19,   255,
     254,   262,     0,   281,   283,   259,   256,     0,   265,   264,
       0,   269,     0,   268,   273,     0,   272,     0,    23,   135,
     134,   438,   437,   445,   411,   412,     0,   439,     0,     0,
     435,   434,     0,   440,     0,     0,     0,   141,   138,   140,
     136,     0,     0,     0,     0,     0,     0,     0,   250,   252,
       0,   107,     0,   236,   238,     0,   289,   287,     0,     0,
       0,   279,     0,     0,   360,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   448,     0,   420,     0,   442,
       0,     0,   417,     0,     0,   432,     0,   391,   390,   389,
     388,   387,   386,   378,   383,   384,   385,   397,   396,   395,
     394,   393,   372,   371,   370,   365,   363,   368,   367,   366,
     369,   374,   373,     0,     0,   403,     0,    27,     0,   447,
       0,     0,   208,     0,   443,     0,   241,   310,   298,     0,
       0,     0,   331,   338,     0,   428,     0,     0,     0,     0,
       0,   395,   267,   267,   277,   276,   275,   414,   413,     0,
     450,   441,     0,   376,   377,     0,   108,     0,     0,     0,
      99,    98,   103,     0,     0,   162,     0,     0,   160,     0,
     172,     0,   193,     0,     0,   191,     0,     0,   216,   217,
     107,   251,   253,     0,     0,   249,     0,   234,     0,   286,
     278,   288,     0,   358,    75,    79,    80,    78,    77,    76,
      74,    73,    72,    71,     0,     0,     0,     0,    53,    56,
      55,   418,   416,    70,   431,     0,     0,   404,     0,     0,
     127,   123,   125,   207,   293,     0,   320,     0,   330,   303,
     299,   300,   329,   320,   341,   340,   243,   426,   261,   260,
     258,   257,   415,     0,    82,     0,   101,     0,     0,     0,
     107,   107,   115,   161,     0,   182,   185,   183,   181,     0,
     179,   176,     0,     0,   192,     0,   205,   206,     0,   202,
       0,     0,   220,   227,   228,     0,     0,   225,     0,   218,
       0,   231,     0,     0,     0,   233,   237,   359,   449,   442,
       0,     0,   241,   245,    54,     0,   430,   429,   405,    28,
       0,     0,     0,   296,   295,   312,     0,     0,     0,     0,
       0,   313,   311,   315,   314,     0,   292,     0,   302,     0,
     333,   334,   336,   335,     0,   332,   451,   102,   106,   105,
      91,     0,     0,   167,   169,     0,   163,   165,     0,   156,
     107,     0,   173,   198,   200,   194,   196,   204,   187,   224,
       0,   222,     0,     0,   212,   247,   239,     0,    57,    58,
     424,     0,   107,   419,   117,   120,     0,     0,     0,     0,
     130,     0,     0,   131,   132,   133,   126,     0,     0,   316,
       0,     0,   323,     0,     0,     0,   308,   309,     0,   305,
     307,   301,     0,   104,   107,     0,   184,   107,     0,   180,
       0,   178,   107,     0,   107,     0,   203,   221,   226,     0,
     235,   423,     0,     0,     0,     0,   143,     0,     0,   147,
       0,     0,   151,     0,     0,   129,   297,     0,   245,     0,
     322,   324,   321,     0,   291,   304,     0,   328,     0,   170,
       0,   166,     0,   201,     0,   197,   223,   422,   421,   118,
     121,   146,   107,   145,   150,   107,   149,   154,   107,   153,
     124,   319,   107,     0,   325,     0,   306,     0,     0,     0,
       0,   318,   326,     0,     0,     0,     0,   144,   148,   152,
     317
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    59,    60,    61,   506,   132,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,   225,    79,    80,    81,   230,    82,
      83,   509,   600,   510,   511,   601,   512,   391,    84,    85,
      86,   232,    87,    88,    89,   651,   652,   721,   722,    90,
      91,    92,   723,   802,   724,   805,   725,   808,    93,   234,
      94,   394,   518,   747,   748,   744,   745,   519,   613,   520,
     692,   609,   610,    95,   235,    96,   395,   525,   754,   755,
     752,   753,   618,   619,    97,    98,   236,    99,   527,   528,
     529,   530,   626,   627,   100,   101,   102,   634,   103,   536,
     104,   343,   344,   238,   401,   402,   239,   240,   105,   106,
     107,   108,   186,   109,   190,   110,   193,   194,   111,   112,
     113,   246,   247,   114,   333,   477,   478,   727,   481,   580,
     581,   668,   738,   739,   576,   662,   663,   778,   664,   665,
     734,   115,   335,   482,   583,   675,   116,   168,   339,   340,
     117,   118,   119,   120,   135,   122,   123,   124,   645,   430,
     556,   643,   125,   169,   345,   126,   127,   128,   155,   201,
     147,   426,   209
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -641
static const yytype_int16 yypact[] =
{
    -641,    88,   819,  -641,    14,  -641,  -641,  -641,  -641,  -641,
    -641,   137,   244,  3111,   165,   248,  3172,   229,  3233,   176,
    3294,  -641,  -641,  3355,   314,  3416,   322,   341,    90,  -641,
    -641,   237,  3477,   387,   404,  3538,   472,   455,   473,   489,
    3599,  5586,   199,  -641,  -641,  -641,  5888,  5462,  5888,   226,
     552,  5888,  5888,  5888,  2989,  5888,  5888,  5888,  5888,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,
    2928,  -641,  -641,  2928,  -641,  -641,  -641,  -641,  2928,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,   232,  2928,   133,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,    35,   233,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,   931,  4053,  -641,   319,  -641,  -641,  -641,  -641,  -641,
    -641,   262,    51,   251,   189,   417,  4113,   311,  -641,   318,
    -641,   343,   195,  4173,   210,   -62,   166,   173,   372,  4343,
     408,   447,  4398,   490,  6388,    54,   499,  -641,  2928,   502,
    4435,   506,  -641,   509,  -641,   512,  -641,  4490,   330,    47,
     513,   521,   528,   530,  6388,   122,   537,   453,   211,  -641,
    -641,   540,  4527,   541,  -641,  -641,    57,   542,   545,   409,
     546,   547,   464,   100,  -641,   551,  -641,   558,  4582,  -641,
    6388,  3050,  -641,  6116,  5694,  -641,   510,  5958,    55,    79,
    6462,    86,   566,  -641,   104,   603,   603,    78,   570,  -641,
     105,    78,    78,    78,    78,  -641,   575,   579,   580,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,   407,  -641,  -641,
    -641,  -641,   585,    83,   588,  -641,   120,    37,   121,  5471,
     586,  5888,  5888,  5888,  5888,  5888,  5888,  5888,  5888,  5888,
    5888,   226,  5728,  -641,  -641,  5791,  5888,  3660,  5888,  5888,
    5888,  5888,  5888,  5888,  5888,  5888,  5888,  5888,   587,  5888,
    5888,  5888,  5888,  5888,  5888,  5888,  5888,  5888,  5888,  5888,
    5888,  5888,  5888,  5888,  -641,  -641,  5579,   589,  -641,  -641,
    -641,   590,  5888,  -641,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  5888,   590,  3721,  -641,
    -641,  -641,  -641,  -641,  -641,   596,  5888,  5888,  -641,  -641,
    -641,  -641,  -641,   190,  -641,   228,  -641,  -641,  -641,   149,
    -641,   597,  -641,   526,  -641,   532,  -641,  -641,  -641,  -641,
    -641,  -641,   333,  -641,  -641,  -641,  -641,  3782,  -641,  -641,
     601,  -641,   602,  -641,  -641,   316,  -641,   606,  -641,  -641,
    -641,  -641,  -641,  6388,  -641,  -641,  6153,  -641,  5825,  5888,
    -641,  -641,   544,  -641,  5888,  5888,  5888,  -641,  -641,  -641,
    -641,  1707,  1374,  1818,   574,   617,  1485,   216,  -641,  -641,
    1929,  -641,  2928,  -641,  -641,    43,  -641,  -641,   607,   611,
     152,  -641,  5888,  6018,  -641,  4619,  4674,  4711,  4766,  4803,
    4858,  4895,  4950,  4987,  5042,   417,   192,  -641,  5922,  5079,
     612,   158,  -641,    58,  5134,  -641,  3943,   267,  6462,  3876,
    3876,  3876,  3876,  3876,  3876,  3876,  3876,  -641,   603,   603,
     603,   603,   699,   699,   781,   461,   461,   221,   221,   221,
     275,    78,    78,  5888,  6078,  -641,   529,  6388,  6190,  -641,
     613,  4233,  -641,  5171,  6388,   619,   618,  -641,   592,   620,
     621,   626,  -641,  -641,   507,  -641,   618,  5888,   627,   630,
     639,   106,  -641,   640,  -641,  -641,  -641,  -641,  -641,  6227,
    6388,  -641,  6277,  3876,  3876,   641,  -641,   396,  3843,   636,
    -641,  -641,  -641,   645,   649,  -641,   481,   411,  -641,   646,
    -641,   653,  -641,   147,   656,  -641,    20,   657,   637,  -641,
    -641,  -641,  -641,   666,  2040,  -641,   614,  -641,   315,  -641,
    -641,  -641,  6314,  -641,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,  -641,  -641,   226,  5888,   171,   631,  -641,  -641,
    -641,  -641,  -641,  -641,  -641,  3904,  6351,  -641,  5888,  5888,
    -641,  -641,  -641,  -641,  -641,   101,    13,   668,  -641,   615,
     598,  -641,  -641,    28,  -641,  -641,  -641,  6388,  -641,  -641,
    -641,  -641,  -641,  5888,  -641,   671,  -641,   672,  5226,   673,
    -641,  -641,  -641,  -641,   389,   604,  -641,  -641,  -641,    84,
    -641,  -641,   674,   406,  -641,   439,  -641,  -641,    89,  -641,
     678,   680,  -641,  -641,  -641,   590,    29,  -641,   683,  -641,
    1596,  -641,   684,   644,   638,  -641,  -641,  -641,   417,  5263,
     159,   689,   618,   232,  -641,   642,  -641,  6425,  -641,  6388,
    4003,   412,  2928,  -641,  -641,  -641,   608,   692,   708,   709,
      21,  -641,  -641,  -641,  -641,   707,  -641,   493,  -641,   621,
    -641,  -641,  -641,  -641,   710,  -641,  6388,  -641,  -641,  -641,
    -641,  2151,  1374,  -641,  -641,   713,  -641,  -641,   531,  -641,
    -641,  2928,  -641,  -641,  -641,  -641,  -641,   422,  -641,  -641,
     715,  -641,   524,   590,  -641,  -641,  -641,   718,  -641,  -641,
    -641,   111,  -641,  -641,  -641,  -641,  5888,   456,   467,   480,
    -641,   714,   412,  -641,  -641,  -641,  -641,   675,  5888,  -641,
     634,   724,  -641,   722,   162,   726,  -641,  -641,   185,  -641,
    -641,  -641,   729,  -641,  -641,  2928,  -641,  -641,  2928,  -641,
    2262,  -641,  -641,  2928,  -641,  2928,  -641,  -641,  -641,   731,
    -641,  -641,   732,  2373,  4293,   733,  -641,  2928,   734,  -641,
    2928,   736,  -641,  2928,   741,  -641,  -641,  5318,   232,  5888,
    -641,  -641,  -641,     4,  -641,  -641,   493,  -641,   930,  -641,
    1041,  -641,  1152,  -641,  1263,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,  -641,  5355,  -641,   743,  -641,  2484,  2595,  2706,
    2817,  -641,  -641,   749,   751,   755,   756,  -641,  -641,  -641,
    -641
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -641,  -641,  -641,  -641,  -641,  -641,     2,  -641,  -641,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,    80,  -641,  -641,  -641,  -641,  -641,  -218,  -641,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,    38,  -641,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,   368,  -641,  -641,
    -641,  -641,    76,  -641,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,  -641,    69,  -641,  -641,  -641,  -641,  -641,  -641,
     239,  -641,  -641,    68,  -641,  -405,  -641,  -641,  -641,  -641,
    -641,  -236,   287,  -640,  -641,  -641,  -641,  -641,  -641,  -641,
    -641,  -641,   737,  -641,  -641,  -641,  -641,   410,  -641,  -641,
    -641,  -105,  -641,  -641,  -641,  -641,  -641,  -641,   298,  -641,
     109,  -641,  -641,    -7,  -641,  -641,   197,  -641,   198,   200,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,   300,
    -641,  -351,    -6,  -641,    -2,    17,   -96,   750,  -641,  -641,
    -641,  -641,  -641,  -641,  -641,  -641,  -641,  -641,   -45,  -641,
    -641,  -641,  -641
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -447
static const yytype_int16 yytable[] =
{
     121,   489,   208,   712,    62,   214,   133,   405,   248,   220,
     814,   145,   392,  -446,   393,   146,   655,   129,   396,   656,
     400,   621,   731,   622,   623,   263,   624,   732,   298,  -446,
     136,   670,   701,   143,   656,   149,   244,   152,  -284,   263,
     154,   245,   160,   245,   537,   167,   409,   211,   341,   174,
     263,   657,   182,   342,   300,   325,   380,   198,   200,   561,
     359,   658,   659,   203,   207,   210,   657,   154,   215,   216,
     217,   154,   221,   222,   223,   224,   658,   659,   121,   298,
     382,   121,   229,   815,   404,   231,   121,   686,     3,   342,
     233,   165,   695,   166,     6,     7,     8,     9,    10,   538,
     733,   121,   653,   366,   702,   241,   660,   388,   390,   591,
    -284,   625,   761,   381,   562,   263,    21,    22,   486,  -241,
     703,   660,  -241,   407,   411,   350,   301,    30,   326,   327,
     327,   687,   360,   327,   242,   265,   696,   383,   812,  -241,
     130,    41,   410,    43,    44,   249,   250,    46,   615,    47,
    -204,   616,   483,   617,   384,   541,   121,   654,  -241,   688,
     329,   560,   709,   265,   697,   782,   137,   762,   138,    48,
      49,   661,   641,   385,   386,   367,   486,   150,   671,   327,
     327,   296,   297,   534,    51,    52,   486,   294,   295,    53,
     243,   475,   303,  -294,  -204,   408,   408,    55,   310,    56,
      57,    58,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   314,   353,   294,   295,   431,   373,   531,
     433,   376,  -204,  -294,   484,   249,   250,   408,   642,   479,
     144,  -298,     8,   327,   327,     8,   304,   783,   170,   245,
     575,   785,   311,   171,   172,   131,   202,   476,   317,   139,
       8,   140,    21,    22,   316,    21,    22,   315,   354,   425,
     786,   480,   141,   532,   318,   299,   413,   554,   415,   416,
     417,   418,   419,   420,   421,   422,   423,   424,   265,   429,
     555,   237,   154,   434,   436,   437,   438,   439,   440,   441,
     442,   443,   444,   445,   446,   466,   448,   449,   450,   451,
     452,   453,   454,   455,   456,   457,   458,   459,   460,   461,
     462,   469,   630,   464,   307,   156,   740,   157,   635,   467,
     494,   308,   495,   161,   265,   291,   292,   293,   162,   263,
     294,   295,   265,   468,   488,   471,   338,     6,     7,   302,
       9,    10,   163,   473,   474,   269,   309,   164,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   158,   636,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,   491,   319,   294,   295,   296,   297,
     292,   293,   681,   682,   294,   295,    43,    44,   176,   121,
     121,   121,   683,   177,   121,   499,   500,   595,   121,   596,
     121,   502,   503,   504,   535,   178,   711,   179,   397,   690,
     398,   321,   611,     4,  -175,     5,     6,     7,     8,     9,
      10,  -128,    12,    13,    14,    15,   616,    16,   617,   542,
      17,   717,   718,   719,    18,   740,   684,    20,    21,    22,
      23,    24,   693,    25,   226,   224,   227,    28,    29,    30,
     322,   180,    32,   691,   399,    35,   187,   765,  -175,   766,
     228,   188,    40,    41,    42,    43,    44,    45,   768,    46,
     769,    47,   750,   183,   191,   184,   249,   250,   185,   192,
     566,   771,   604,   772,   362,   605,   694,   606,   607,   608,
     195,    48,    49,   324,   763,   196,    50,     6,     7,   736,
       9,    10,   328,   767,   587,   330,    51,    52,   584,   332,
     640,    53,   334,   338,   770,   336,   346,    54,   265,    55,
     737,    56,    57,    58,   347,   598,   788,   773,   623,   790,
     624,   348,   121,   349,   792,   605,   794,   606,   607,   608,
     351,   352,   263,   355,   358,   361,    43,    44,  -270,   363,
     364,   365,   638,   212,   368,   213,     6,     7,     8,     9,
      10,   369,   288,   289,   290,   291,   292,   293,   377,   387,
     294,   295,   639,   389,   224,   514,   161,   515,    21,    22,
     163,   195,   647,  -171,   817,   649,   650,   818,   403,    30,
     819,   406,   414,   447,   820,   465,     8,   516,   517,   472,
     485,   486,   501,    41,   487,    43,    44,   492,   493,    46,
     676,    47,   192,   539,   540,   559,   570,   568,   521,   700,
     522,  -174,   574,   578,   342,   480,  -171,   579,   121,   582,
     588,    48,    49,   589,   644,     6,     7,     8,     9,    10,
     523,   517,   590,  -271,   594,   599,    51,    52,   602,   121,
     121,    53,   603,   720,   726,   612,   614,    21,    22,    55,
     265,    56,    57,    58,  -174,   620,   628,   526,    30,   631,
     633,   666,   667,   669,   677,   678,   680,   689,   685,   121,
     121,   698,    41,   699,    43,    44,   704,   705,    46,   121,
      47,   706,   710,   751,   707,   729,   728,   759,   713,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
      48,    49,   294,   295,   196,   730,   735,   746,   757,   742,
     121,   760,   779,   774,   720,    51,    52,   780,   781,   784,
      53,   776,   787,   764,   796,   797,   801,   804,    55,   807,
      56,    57,    58,   121,   810,   777,   121,   789,   121,   822,
     791,   121,   827,   121,   828,   793,   265,   795,   829,   830,
     775,   121,   743,   524,   749,   121,   756,   629,   121,   803,
     758,   121,   806,   586,   189,   809,   577,   496,   741,   816,
     672,   673,   175,   674,   585,     0,   121,     0,   121,     0,
     121,     0,   121,     0,     0,     0,   813,   285,   286,   287,
     288,   289,   290,   291,   292,   293,     0,     0,   294,   295,
       0,     0,     0,     0,     0,   121,   121,   121,   121,    -2,
       4,     0,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,     0,    16,     0,     0,    17,   265,     0,
       0,    18,    19,     0,    20,    21,    22,    23,    24,     0,
      25,    26,     0,    27,    28,    29,    30,     0,    31,    32,
      33,    34,    35,    36,    37,    38,     0,    39,     0,    40,
      41,    42,    43,    44,    45,     0,    46,     0,    47,     0,
     286,   287,   288,   289,   290,   291,   292,   293,     0,     0,
     294,   295,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,    50,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    51,    52,     0,     0,     0,    53,     0,
       0,     0,     0,     0,    54,     0,    55,     0,    56,    57,
      58,     4,     0,     5,     6,     7,     8,     9,    10,  -168,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,  -168,  -168,    20,    21,    22,    23,    24,
       0,    25,   226,     0,   227,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,  -168,   228,     0,
      40,    41,    42,    43,    44,    45,     0,    46,     0,    47,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,     0,     0,     0,     0,   261,     0,     0,    48,
      49,     0,     0,     0,    50,     0,     0,     0,     0,   262,
       0,     0,     0,     0,    51,    52,     0,     0,     0,    53,
       0,     0,     0,     0,     0,    54,     0,    55,     0,    56,
      57,    58,     4,     0,     5,     6,     7,     8,     9,    10,
    -164,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,  -164,  -164,    20,    21,    22,    23,
      24,     0,    25,   226,     0,   227,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,  -164,   228,
       0,    40,    41,    42,    43,    44,    45,     0,    46,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    51,    52,     0,     0,     0,
      53,     0,     0,     0,     0,     0,    54,     0,    55,     0,
      56,    57,    58,     4,     0,     5,     6,     7,     8,     9,
      10,  -199,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,  -199,  -199,    20,    21,    22,
      23,    24,     0,    25,   226,     0,   227,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,  -199,
     228,     0,    40,    41,    42,    43,    44,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,    49,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    51,    52,     0,     0,
       0,    53,     0,     0,     0,     0,     0,    54,     0,    55,
       0,    56,    57,    58,     4,     0,     5,     6,     7,     8,
       9,    10,  -195,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,  -195,  -195,    20,    21,
      22,    23,    24,     0,    25,   226,     0,   227,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
    -195,   228,     0,    40,    41,    42,    43,    44,    45,     0,
      46,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    51,    52,     0,
       0,     0,    53,     0,     0,     0,     0,     0,    54,     0,
      55,     0,    56,    57,    58,     4,     0,     5,     6,     7,
       8,     9,    10,   -97,    12,    13,    14,    15,     0,    16,
     507,   508,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   226,     0,   227,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   228,     0,    40,    41,    42,    43,    44,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,    49,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    51,    52,
       0,     0,     0,    53,     0,     0,     0,     0,     0,    54,
       0,    55,     0,    56,    57,    58,     4,     0,     5,     6,
       7,     8,     9,    10,  -215,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,   526,    25,   226,     0,   227,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   228,     0,    40,    41,    42,    43,    44,
      45,     0,    46,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    51,
      52,     0,     0,     0,    53,     0,     0,     0,     0,     0,
      54,     0,    55,     0,    56,    57,    58,     4,     0,     5,
       6,     7,     8,     9,    10,  -219,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,  -219,    25,   226,     0,
     227,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   228,     0,    40,    41,    42,    43,
      44,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,    49,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      51,    52,     0,     0,     0,    53,     0,     0,     0,     0,
       0,    54,     0,    55,     0,    56,    57,    58,     4,     0,
       5,     6,     7,     8,     9,    10,   505,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   226,
       0,   227,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,     0,   228,     0,    40,    41,    42,
      43,    44,    45,     0,    46,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    51,    52,     0,     0,     0,    53,     0,     0,     0,
       0,     0,    54,     0,    55,     0,    56,    57,    58,     4,
       0,     5,     6,     7,     8,     9,    10,   513,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     226,     0,   227,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   228,     0,    40,    41,
      42,    43,    44,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,    49,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    51,    52,     0,     0,     0,    53,     0,     0,
       0,     0,     0,    54,     0,    55,     0,    56,    57,    58,
       4,     0,     5,     6,     7,     8,     9,    10,   533,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   226,     0,   227,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,     0,   228,     0,    40,
      41,    42,    43,    44,    45,     0,    46,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,    50,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    51,    52,     0,     0,     0,    53,     0,
       0,     0,     0,     0,    54,     0,    55,     0,    56,    57,
      58,     4,     0,     5,     6,     7,     8,     9,    10,   632,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   226,     0,   227,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   228,     0,
      40,    41,    42,    43,    44,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
      49,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    51,    52,     0,     0,     0,    53,
       0,     0,     0,     0,     0,    54,     0,    55,     0,    56,
      57,    58,     4,     0,     5,     6,     7,     8,     9,    10,
    -100,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   226,     0,   227,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   228,
       0,    40,    41,    42,    43,    44,    45,     0,    46,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    51,    52,     0,     0,     0,
      53,     0,     0,     0,     0,     0,    54,     0,    55,     0,
      56,    57,    58,     4,     0,     5,     6,     7,     8,     9,
      10,  -177,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   226,     0,   227,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     228,     0,    40,    41,    42,    43,    44,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,    49,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    51,    52,     0,     0,
       0,    53,     0,     0,     0,     0,     0,    54,     0,    55,
       0,    56,    57,    58,     4,     0,     5,     6,     7,     8,
       9,    10,   798,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   226,     0,   227,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
       0,   228,     0,    40,    41,    42,    43,    44,    45,     0,
      46,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    51,    52,     0,
       0,     0,    53,     0,     0,     0,     0,     0,    54,     0,
      55,     0,    56,    57,    58,     4,     0,     5,     6,     7,
       8,     9,    10,   823,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   226,     0,   227,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   228,     0,    40,    41,    42,    43,    44,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,    49,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    51,    52,
       0,     0,     0,    53,     0,     0,     0,     0,     0,    54,
       0,    55,     0,    56,    57,    58,     4,     0,     5,     6,
       7,     8,     9,    10,   824,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   226,     0,   227,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   228,     0,    40,    41,    42,    43,    44,
      45,     0,    46,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    51,
      52,     0,     0,     0,    53,     0,     0,     0,     0,     0,
      54,     0,    55,     0,    56,    57,    58,     4,     0,     5,
       6,     7,     8,     9,    10,   825,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   226,     0,
     227,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   228,     0,    40,    41,    42,    43,
      44,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,    49,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      51,    52,     0,     0,     0,    53,     0,     0,     0,     0,
       0,    54,     0,    55,     0,    56,    57,    58,     4,     0,
       5,     6,     7,     8,     9,    10,   826,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   226,
       0,   227,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,     0,   228,     0,    40,    41,    42,
      43,    44,    45,     0,    46,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    51,    52,     0,     0,     0,    53,     0,     0,     0,
       0,     0,    54,     0,    55,     0,    56,    57,    58,     4,
       0,     5,     6,     7,     8,     9,    10,     0,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     226,     0,   227,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   228,     0,    40,    41,
      42,    43,    44,    45,     0,    46,     0,    47,     0,     0,
     218,     0,   219,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,    49,     0,
       0,     0,    50,     0,     0,    21,    22,     0,     0,     0,
       0,     0,    51,    52,     0,     0,    30,    53,     0,     0,
       0,     0,     0,    54,     0,    55,     0,    56,    57,    58,
      41,     0,    43,    44,     0,     0,    46,     0,    47,     0,
       0,   371,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,    51,    52,     0,     0,    30,    53,     0,
       0,     0,     0,     0,     0,     0,    55,     0,    56,    57,
      58,    41,     0,    43,    44,     0,     0,    46,   372,    47,
       0,     0,   134,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
      49,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,     0,     0,     0,    51,    52,     0,     0,    30,    53,
       0,     0,     0,     0,     0,     0,     0,    55,     0,    56,
      57,    58,    41,     0,    43,    44,     0,     0,    46,     0,
      47,     0,     0,   142,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,    51,    52,     0,     0,    30,
      53,     0,     0,     0,     0,     0,     0,     0,    55,     0,
      56,    57,    58,    41,     0,    43,    44,     0,     0,    46,
       0,    47,     0,     0,   148,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,    49,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,    51,    52,     0,     0,
      30,    53,     0,     0,     0,     0,     0,     0,     0,    55,
       0,    56,    57,    58,    41,     0,    43,    44,     0,     0,
      46,     0,    47,     0,     0,   151,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,     0,     0,     0,
      21,    22,     0,     0,     0,     0,     0,    51,    52,     0,
       0,    30,    53,     0,     0,     0,     0,     0,     0,     0,
      55,     0,    56,    57,    58,    41,     0,    43,    44,     0,
       0,    46,     0,    47,     0,     0,   153,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,    49,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,    51,    52,
       0,     0,    30,    53,     0,     0,     0,     0,     0,     0,
       0,    55,     0,    56,    57,    58,    41,     0,    43,    44,
       0,     0,    46,     0,    47,     0,     0,   159,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,     0,     0,     0,     0,
       0,     0,    21,    22,     0,     0,     0,     0,     0,    51,
      52,     0,     0,    30,    53,     0,     0,     0,     0,     0,
       0,     0,    55,     0,    56,    57,    58,    41,     0,    43,
      44,     0,     0,    46,     0,    47,     0,     0,   173,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,    49,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
      51,    52,     0,     0,    30,    53,     0,     0,     0,     0,
       0,     0,     0,    55,     0,    56,    57,    58,    41,     0,
      43,    44,     0,     0,    46,     0,    47,     0,     0,   181,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,    49,     0,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,     0,
       0,    51,    52,     0,     0,    30,    53,     0,     0,     0,
       0,     0,     0,     0,    55,     0,    56,    57,    58,    41,
       0,    43,    44,     0,     0,    46,     0,    47,     0,     0,
     197,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,    49,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
       0,     0,    51,    52,     0,     0,    30,    53,     0,     0,
       0,     0,     0,     0,     0,    55,     0,    56,    57,    58,
      41,     0,    43,    44,     0,     0,    46,     0,    47,     0,
       0,   435,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,    51,    52,     0,     0,    30,    53,     0,
       0,     0,     0,     0,     0,     0,    55,     0,    56,    57,
      58,    41,     0,    43,    44,     0,     0,    46,     0,    47,
       0,     0,   470,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
      49,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,     0,     0,     0,    51,    52,     0,     0,    30,    53,
       0,     0,     0,     0,     0,     0,     0,    55,     0,    56,
      57,    58,    41,     0,    43,    44,     0,     0,    46,     0,
      47,     0,     0,   490,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,    51,    52,     0,     0,    30,
      53,     0,     0,     0,     0,     0,     0,     0,    55,     0,
      56,    57,    58,    41,     0,    43,    44,     0,     0,    46,
       0,    47,     0,     0,   597,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,    49,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,    51,    52,     0,     0,
      30,    53,     0,     0,     0,     0,     0,     0,     0,    55,
       0,    56,    57,    58,    41,     0,    43,    44,     0,     0,
      46,     0,    47,     0,     0,   646,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,     0,     0,     0,     0,     0,     0,
      21,    22,     0,   265,     0,     0,     0,    51,    52,     0,
       0,    30,    53,     0,   564,     0,     0,     0,     0,     0,
      55,     0,    56,    57,    58,    41,     0,    43,    44,     0,
       0,    46,     0,    47,     0,   278,   279,   280,   281,   282,
       0,     0,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,    48,    49,   294,   295,     0,     0,     0,
     565,     0,     0,     0,     0,     0,     0,     0,    51,    52,
     265,     0,     0,    53,     0,     0,   714,     0,     0,     0,
       0,    55,     0,    56,    57,    58,     0,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,     0,     0,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     715,     0,   294,   295,     0,     0,   264,     0,     0,     0,
     265,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   716,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,     0,     0,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     265,     0,   294,   295,     0,     0,   305,     0,     0,     0,
       0,     0,     0,     0,   266,     0,     0,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,     0,     0,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     306,     0,   294,   295,     0,     0,     0,     0,     0,     0,
     265,     0,     0,     0,     0,     0,   312,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,     0,     0,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     313,     0,   294,   295,     0,     0,     0,     0,     0,     0,
     265,     0,     0,     0,     0,     0,   571,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,     0,     0,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     572,     0,   294,   295,     0,     0,     0,     0,     0,     0,
     265,     0,     0,     0,     0,     0,   799,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,     0,     0,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     800,     0,   294,   295,     0,     0,   320,     0,     0,     0,
     265,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,     0,     0,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     265,   323,   294,   295,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,     0,   331,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
       0,     0,   294,   295,     0,   265,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   267,   268,   269,     0,     0,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   265,   337,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,     0,     0,   294,   295,     0,
       0,   267,   268,   269,     0,     0,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,     0,
     356,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,     0,     0,   294,   295,     0,   265,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   267,   268,   269,     0,
       0,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   265,   370,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,     0,     0,   294,
     295,     0,     0,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   357,   281,
     282,     0,   544,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,     0,     0,   294,   295,     0,   265,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   265,   545,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,     0,
       0,   294,   295,     0,     0,   267,   268,   269,     0,     0,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,     0,   546,   283,   284,   285,   286,   287,
     288,   289,   290,   291,   292,   293,     0,     0,   294,   295,
       0,   265,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     267,   268,   269,     0,     0,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   265,   547,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,     0,     0,   294,   295,     0,     0,   267,   268,   269,
       0,     0,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,     0,   548,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,     0,     0,
     294,   295,     0,   265,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   267,   268,   269,     0,     0,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     265,   549,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,     0,     0,   294,   295,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,     0,   550,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
       0,     0,   294,   295,     0,   265,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   267,   268,   269,     0,     0,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   265,   551,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,     0,     0,   294,   295,     0,
       0,   267,   268,   269,     0,     0,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,     0,
     552,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,     0,     0,   294,   295,     0,   265,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   267,   268,   269,     0,
       0,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   265,   553,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,     0,     0,   294,
     295,     0,     0,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,     0,   558,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,     0,     0,   294,   295,     0,   265,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   267,   268,
     269,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   265,   563,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,     0,
       0,   294,   295,     0,     0,   267,   268,   269,     0,     0,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,     0,   573,   283,   284,   285,   286,   287,
     288,   289,   290,   291,   292,   293,     0,     0,   294,   295,
       0,   265,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     267,   268,   269,     0,     0,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   265,   679,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,     0,     0,   294,   295,     0,     0,   267,   268,   269,
       0,     0,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,     0,   708,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,     0,     0,
     294,   295,     0,   265,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   267,   268,   269,     0,     0,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     265,   811,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,     0,     0,   294,   295,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,     0,   821,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
       0,     0,   294,   295,     0,   265,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   267,   268,   269,     0,     0,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   265,     0,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,     0,     0,   294,   295,     0,
       0,   267,   268,   269,     0,     0,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,     0,
       0,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,     0,     0,   294,   295,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,     0,     0,    21,    22,    30,
       0,     0,     0,     0,     0,     0,     0,     0,    30,   204,
       0,     0,     0,    41,     0,    43,    44,     0,   204,    46,
     205,    47,    41,     0,    43,    44,     0,     0,    46,     0,
      47,     0,     0,     0,   206,     0,     0,     0,     0,     0,
       0,    48,    49,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,     0,    51,    52,     0,     0,
       0,    53,     0,     0,     0,    51,    52,     0,     0,    55,
      53,    56,    57,    58,   412,     0,     0,     0,    55,     0,
      56,    57,    58,     6,     7,     8,     9,    10,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,     0,
       0,     0,    21,    22,     0,     0,    30,     0,     0,     0,
       0,     0,     0,    30,     0,     0,   204,     0,     0,     0,
      41,     0,    43,    44,     0,     0,    46,    41,    47,    43,
      44,     0,     0,    46,   199,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
       0,     0,     0,     0,     0,    48,    49,     0,     0,     0,
       0,     0,     0,    51,    52,     0,     0,     0,    53,     0,
      51,    52,   463,     0,     0,    53,    55,     0,    56,    57,
      58,     0,     0,    55,     0,    56,    57,    58,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    30,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,    41,     0,    43,    44,     0,
       0,    46,   375,    47,    21,    22,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    30,     0,     0,     0,     0,
       0,     0,     0,    48,    49,     0,   427,     0,     0,    41,
       0,    43,    44,     0,     0,    46,     0,    47,    51,    52,
       0,     0,     0,    53,     0,     6,     7,     8,     9,    10,
       0,    55,     0,    56,    57,    58,     0,    48,    49,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,     0,    51,    52,     0,     0,     0,    53,    30,     6,
       7,     8,     9,    10,     0,    55,     0,    56,    57,   428,
       0,     0,    41,     0,    43,    44,     0,   432,    46,     0,
      47,    21,    22,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    30,     0,     0,     0,     0,     0,     0,     0,
      48,    49,     0,     0,     0,     0,    41,     0,    43,    44,
       0,     0,    46,   498,    47,    51,    52,     0,     0,     0,
      53,     0,     6,     7,     8,     9,    10,     0,    55,     0,
      56,    57,    58,     0,    48,    49,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,     0,     0,     0,    51,
      52,     0,     0,     0,    53,    30,     6,     7,     8,     9,
      10,     0,    55,     0,    56,    57,    58,     0,     0,    41,
       0,    43,    44,     0,     0,    46,     0,    47,    21,    22,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    30,
       0,     0,     0,     0,     0,     0,     0,    48,    49,     0,
       0,     0,     0,    41,     0,    43,    44,     0,     0,    46,
       0,    47,    51,    52,     0,     0,     0,    53,     0,     0,
       0,     0,     0,     0,     0,    55,     0,    56,    57,    58,
       0,    48,    49,     0,     0,   378,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   265,    51,    52,     0,     0,
       0,    53,     0,     0,     0,     0,     0,     0,     0,    55,
     379,    56,    57,   557,   267,   268,   269,     0,     0,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,     0,     0,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,   378,     0,   294,   295,     0,
       0,     0,     0,     0,     0,   265,   543,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   267,   268,   269,     0,     0,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,     0,     0,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,   378,     0,   294,   295,     0,
       0,     0,     0,     0,     0,   265,   567,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   267,   268,   269,     0,     0,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   374,   265,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,     0,     0,   294,   295,     0,
       0,     0,   267,   268,   269,     0,     0,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     265,   497,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,     0,     0,   294,   295,     0,     0,   267,
     268,   269,     0,     0,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   265,     0,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
       0,     0,   294,   295,   569,     0,   267,   268,   269,     0,
       0,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   265,   592,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,     0,     0,   294,
     295,     0,     0,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,     0,     0,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,   265,     0,   294,   295,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   593,
       0,     0,     0,   267,   268,   269,     0,     0,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   265,   637,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,     0,     0,   294,   295,     0,     0,
     267,   268,   269,     0,     0,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   265,   648,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,     0,     0,   294,   295,     0,     0,   267,   268,   269,
       0,     0,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   265,     0,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,     0,     0,
     294,   295,     0,     0,   267,   268,   269,     0,     0,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   265,     0,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,     0,     0,   294,   295,     0,
       0,     0,   268,   269,     0,     0,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   265,
       0,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,     0,     0,   294,   295,     0,     0,     0,     0,
       0,     0,     0,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,     0,     0,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,     0,
       0,   294,   295
};

static const yytype_int16 yycheck[] =
{
       2,   352,    47,   643,     2,    50,    12,   243,   113,    54,
       6,    17,   230,    75,   232,    17,     3,     3,   236,     6,
     238,     1,     1,     3,     4,   121,     6,     6,   124,    91,
      13,     3,     3,    16,     6,    18,     1,    20,     3,   135,
      23,     6,    25,     6,     1,    28,     9,    49,     1,    32,
     146,    38,    35,     6,     3,     1,     1,    40,    41,     1,
       3,    48,    49,    46,    47,    48,    38,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    48,    49,    80,   175,
       1,    83,    80,    79,     1,    83,    88,     3,     0,     6,
      88,     1,     3,     3,     4,     5,     6,     7,     8,    56,
      79,   103,     1,     3,    75,   103,    93,     3,     3,     3,
      75,    91,     1,    58,    56,   211,    26,    27,    75,    72,
      91,    93,    75,     3,     3,     3,    75,    37,    74,    75,
      75,    47,    75,    75,     1,    57,    47,    58,   778,    56,
       3,    51,   247,    53,    54,    59,    60,    57,     1,    59,
       3,     4,     3,     6,    75,     3,   158,    56,    75,    75,
     158,     3,     3,    57,    75,     3,     1,    56,     3,    79,
      80,   576,     1,    87,    88,    75,    75,     1,   583,    75,
      75,    59,    60,   401,    94,    95,    75,   109,   110,    99,
      57,     1,     3,     3,    47,    75,    75,   107,     3,   109,
     110,   111,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,     3,     3,   109,   110,   262,   201,     3,
     265,   204,    75,    33,    75,    59,    60,    75,    57,     1,
       1,     3,     6,    75,    75,     6,    47,    75,     1,     6,
     476,    56,    47,     6,     7,     1,    47,    57,    75,     1,
       6,     3,    26,    27,    88,    26,    27,    47,    47,   261,
      75,    33,    14,    47,    91,     3,   249,    75,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,    57,   262,
      88,    49,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   301,   279,   280,   281,   282,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,   317,   530,   296,     3,     1,   667,     3,     3,   302,
       4,     3,     6,     1,    57,   104,   105,   106,     6,   425,
     109,   110,    57,   316,     1,   318,     6,     4,     5,    88,
       7,     8,     1,   326,   327,    78,     3,     6,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    47,    47,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   357,     3,   109,   110,    59,    60,
     105,   106,   600,   601,   109,   110,    53,    54,     1,   391,
     392,   393,     3,     6,   396,   378,   379,     1,   400,     3,
     402,   384,   385,   386,   402,     1,   642,     3,     1,     3,
       3,     3,     1,     1,     3,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,     4,    15,     6,   412,
      18,    19,    20,    21,    22,   786,    47,    25,    26,    27,
      28,    29,     3,    31,    32,   428,    34,    35,    36,    37,
       3,    47,    40,    47,    47,    43,     1,     1,    47,     3,
      48,     6,    50,    51,    52,    53,    54,    55,     1,    57,
       3,    59,   690,     1,     1,     3,    59,    60,     6,     6,
     463,     1,     1,     3,    75,     4,    47,     6,     7,     8,
       1,    79,    80,     3,   712,     6,    84,     4,     5,     6,
       7,     8,     3,    47,   487,     3,    94,    95,     1,     3,
     555,    99,     3,     6,    47,     3,     3,   105,    57,   107,
      27,   109,   110,   111,     3,   508,   744,    47,     4,   747,
       6,     3,   534,     3,   752,     4,   754,     6,     7,     8,
       3,    88,   638,     3,     3,     3,    53,    54,     3,     3,
       3,    87,   554,     1,     3,     3,     4,     5,     6,     7,
       8,     3,   101,   102,   103,   104,   105,   106,    58,     3,
     109,   110,   555,     3,   557,     1,     1,     3,    26,    27,
       1,     1,   565,     9,   802,   568,   569,   805,     3,    37,
     808,     3,     6,     6,   812,     6,     6,    23,    24,     3,
       3,    75,    58,    51,    72,    53,    54,     6,     6,    57,
     593,    59,     6,     6,     3,     3,     3,    88,     1,   625,
       3,    47,     3,     3,     6,    33,     9,     6,   630,     3,
       3,    79,    80,     3,     3,     4,     5,     6,     7,     8,
      23,    24,     3,     3,     3,     9,    94,    95,     3,   651,
     652,    99,     3,   651,   652,     9,     3,    26,    27,   107,
      57,   109,   110,   111,    47,     9,     9,    30,    37,     3,
      56,     3,    57,    75,     3,     3,     3,     3,    74,   681,
     682,     3,    51,     3,    53,    54,     3,     3,    57,   691,
      59,    47,     3,   691,    56,     3,    88,   703,    56,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      79,    80,   109,   110,     6,     6,     9,     4,     3,     9,
     722,     3,    88,     9,   722,    94,    95,     3,     6,     3,
      99,    56,     3,   716,     3,     3,     3,     3,   107,     3,
     109,   110,   111,   745,     3,   728,   748,   745,   750,     6,
     748,   753,     3,   755,     3,   753,    57,   755,     3,     3,
     722,   763,   682,   395,   688,   767,   697,   528,   770,   767,
     702,   773,   770,   486,    37,   773,   478,   367,   669,   786,
     583,   583,    32,   583,   484,    -1,   788,    -1,   790,    -1,
     792,    -1,   794,    -1,    -1,    -1,   779,    98,    99,   100,
     101,   102,   103,   104,   105,   106,    -1,    -1,   109,   110,
      -1,    -1,    -1,    -1,    -1,   817,   818,   819,   820,     0,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    57,    -1,
      -1,    22,    23,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    39,    40,
      41,    42,    43,    44,    45,    46,    -1,    48,    -1,    50,
      51,    52,    53,    54,    55,    -1,    57,    -1,    59,    -1,
      99,   100,   101,   102,   103,   104,   105,   106,    -1,    -1,
     109,   110,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,
      -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,
      -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,   110,
     111,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    47,    48,    -1,
      50,    51,    52,    53,    54,    55,    -1,    57,    -1,    59,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    -1,    -1,    -1,    -1,    75,    -1,    -1,    79,
      80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    88,
      -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,
     110,   111,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    23,    24,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    47,    48,
      -1,    50,    51,    52,    53,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,
      99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,
     109,   110,   111,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    24,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    47,
      48,    -1,    50,    51,    52,    53,    54,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    79,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,
      -1,   109,   110,   111,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    23,    24,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      47,    48,    -1,    50,    51,    52,    53,    54,    55,    -1,
      57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    -1,    -1,    -1,    84,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,
      -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,
     107,    -1,   109,   110,   111,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      16,    17,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    54,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    84,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,
      -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,
      -1,   107,    -1,   109,   110,   111,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    30,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    84,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,
      95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,    -1,   109,   110,   111,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    30,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,
      84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,
      -1,   105,    -1,   107,    -1,   109,   110,   111,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,
      53,    54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,
      -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,
      -1,    -1,   105,    -1,   107,    -1,   109,   110,   111,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    54,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,
      -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,
      -1,    -1,    -1,   105,    -1,   107,    -1,   109,   110,   111,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,
      51,    52,    53,    54,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,
      -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,
      -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,   110,
     111,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    54,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,
      80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,
     110,   111,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    54,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,
      99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,
     109,   110,   111,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    54,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    79,    80,    -1,    -1,    -1,    84,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,
      -1,   109,   110,   111,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      -1,    48,    -1,    50,    51,    52,    53,    54,    55,    -1,
      57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    -1,    -1,    -1,    84,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,
      -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,
     107,    -1,   109,   110,   111,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    54,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    84,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,
      -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,
      -1,   107,    -1,   109,   110,   111,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    84,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,
      95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,    -1,   109,   110,   111,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,
      84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,
      -1,   105,    -1,   107,    -1,   109,   110,   111,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,
      53,    54,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,
      -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,
      -1,    -1,   105,    -1,   107,    -1,   109,   110,   111,     1,
      -1,     3,     4,     5,     6,     7,     8,    -1,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    54,    55,    -1,    57,    -1,    59,    -1,    -1,
       1,    -1,     3,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,
      -1,    -1,    84,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    -1,    94,    95,    -1,    -1,    37,    99,    -1,    -1,
      -1,    -1,    -1,   105,    -1,   107,    -1,   109,   110,   111,
      51,    -1,    53,    54,    -1,    -1,    57,    -1,    59,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    94,    95,    -1,    -1,    37,    99,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   107,    -1,   109,   110,
     111,    51,    -1,    53,    54,    -1,    -1,    57,    58,    59,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,
      80,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    37,    99,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,    -1,   109,
     110,   111,    51,    -1,    53,    54,    -1,    -1,    57,    -1,
      59,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    37,
      99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,    -1,
     109,   110,   111,    51,    -1,    53,    54,    -1,    -1,    57,
      -1,    59,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      37,    99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,
      -1,   109,   110,   111,    51,    -1,    53,    54,    -1,    -1,
      57,    -1,    59,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,
      -1,    37,    99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     107,    -1,   109,   110,   111,    51,    -1,    53,    54,    -1,
      -1,    57,    -1,    59,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    94,    95,
      -1,    -1,    37,    99,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   107,    -1,   109,   110,   111,    51,    -1,    53,    54,
      -1,    -1,    57,    -1,    59,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    94,
      95,    -1,    -1,    37,    99,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   107,    -1,   109,   110,   111,    51,    -1,    53,
      54,    -1,    -1,    57,    -1,    59,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      94,    95,    -1,    -1,    37,    99,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   107,    -1,   109,   110,   111,    51,    -1,
      53,    54,    -1,    -1,    57,    -1,    59,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,
      -1,    94,    95,    -1,    -1,    37,    99,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   107,    -1,   109,   110,   111,    51,
      -1,    53,    54,    -1,    -1,    57,    -1,    59,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    -1,    94,    95,    -1,    -1,    37,    99,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   107,    -1,   109,   110,   111,
      51,    -1,    53,    54,    -1,    -1,    57,    -1,    59,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    94,    95,    -1,    -1,    37,    99,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   107,    -1,   109,   110,
     111,    51,    -1,    53,    54,    -1,    -1,    57,    -1,    59,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,
      80,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    37,    99,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,    -1,   109,
     110,   111,    51,    -1,    53,    54,    -1,    -1,    57,    -1,
      59,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    37,
      99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,    -1,
     109,   110,   111,    51,    -1,    53,    54,    -1,    -1,    57,
      -1,    59,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      37,    99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,
      -1,   109,   110,   111,    51,    -1,    53,    54,    -1,    -1,
      57,    -1,    59,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    57,    -1,    -1,    -1,    94,    95,    -1,
      -1,    37,    99,    -1,     1,    -1,    -1,    -1,    -1,    -1,
     107,    -1,   109,   110,   111,    51,    -1,    53,    54,    -1,
      -1,    57,    -1,    59,    -1,    89,    90,    91,    92,    93,
      -1,    -1,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    79,    80,   109,   110,    -1,    -1,    -1,
      47,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,
      57,    -1,    -1,    99,    -1,    -1,     3,    -1,    -1,    -1,
      -1,   107,    -1,   109,   110,   111,    -1,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      47,    -1,   109,   110,    -1,    -1,     3,    -1,    -1,    -1,
      57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      57,    -1,   109,   110,    -1,    -1,     3,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    -1,    -1,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      47,    -1,   109,   110,    -1,    -1,    -1,    -1,    -1,    -1,
      57,    -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      47,    -1,   109,   110,    -1,    -1,    -1,    -1,    -1,    -1,
      57,    -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      47,    -1,   109,   110,    -1,    -1,    -1,    -1,    -1,    -1,
      57,    -1,    -1,    -1,    -1,    -1,     3,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      47,    -1,   109,   110,    -1,    -1,     3,    -1,    -1,    -1,
      57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      57,     3,   109,   110,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,     3,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      -1,    -1,   109,   110,    -1,    57,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    57,     3,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    -1,    -1,   109,   110,    -1,
      -1,    76,    77,    78,    -1,    -1,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    -1,
       3,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,    -1,    -1,   109,   110,    -1,    57,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    78,    -1,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    57,     3,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,    -1,    -1,   109,
     110,    -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    -1,     3,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    -1,    -1,   109,   110,    -1,    57,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    57,     3,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,   109,   110,    -1,    -1,    76,    77,    78,    -1,    -1,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    -1,     3,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,    -1,    -1,   109,   110,
      -1,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      76,    77,    78,    -1,    -1,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    57,     3,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,    -1,    -1,   109,   110,    -1,    -1,    76,    77,    78,
      -1,    -1,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    -1,     3,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,    -1,    -1,
     109,   110,    -1,    57,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      57,     3,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    -1,    -1,   109,   110,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,     3,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      -1,    -1,   109,   110,    -1,    57,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    57,     3,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    -1,    -1,   109,   110,    -1,
      -1,    76,    77,    78,    -1,    -1,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    -1,
       3,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,    -1,    -1,   109,   110,    -1,    57,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    76,    77,    78,    -1,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    57,     3,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,    -1,    -1,   109,
     110,    -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    -1,     3,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    -1,    -1,   109,   110,    -1,    57,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    77,
      78,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    57,     3,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,   109,   110,    -1,    -1,    76,    77,    78,    -1,    -1,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    -1,     3,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,    -1,    -1,   109,   110,
      -1,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      76,    77,    78,    -1,    -1,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    57,     3,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,    -1,    -1,   109,   110,    -1,    -1,    76,    77,    78,
      -1,    -1,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    -1,     3,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,    -1,    -1,
     109,   110,    -1,    57,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      57,     3,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    -1,    -1,   109,   110,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,     3,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      -1,    -1,   109,   110,    -1,    57,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    57,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    -1,    -1,   109,   110,    -1,
      -1,    76,    77,    78,    -1,    -1,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    -1,
      -1,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,    -1,    -1,   109,   110,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    37,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    47,
      -1,    -1,    -1,    51,    -1,    53,    54,    -1,    47,    57,
      58,    59,    51,    -1,    53,    54,    -1,    -1,    57,    -1,
      59,    -1,    -1,    -1,    72,    -1,    -1,    -1,    -1,    -1,
      -1,    79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    94,    95,    -1,    -1,   107,
      99,   109,   110,   111,   103,    -1,    -1,    -1,   107,    -1,
     109,   110,   111,     4,     5,     6,     7,     8,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    37,    -1,    -1,    -1,
      -1,    -1,    -1,    37,    -1,    -1,    47,    -1,    -1,    -1,
      51,    -1,    53,    54,    -1,    -1,    57,    51,    59,    53,
      54,    -1,    -1,    57,    58,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,
      -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    -1,
      -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,
      94,    95,   103,    -1,    -1,    99,   107,    -1,   109,   110,
     111,    -1,    -1,   107,    -1,   109,   110,   111,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    37,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    51,    -1,    53,    54,    -1,
      -1,    57,    58,    59,    26,    27,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    79,    80,    -1,    48,    -1,    -1,    51,
      -1,    53,    54,    -1,    -1,    57,    -1,    59,    94,    95,
      -1,    -1,    -1,    99,    -1,     4,     5,     6,     7,     8,
      -1,   107,    -1,   109,   110,   111,    -1,    79,    80,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    -1,    94,    95,    -1,    -1,    -1,    99,    37,     4,
       5,     6,     7,     8,    -1,   107,    -1,   109,   110,   111,
      -1,    -1,    51,    -1,    53,    54,    -1,    56,    57,    -1,
      59,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    80,    -1,    -1,    -1,    -1,    51,    -1,    53,    54,
      -1,    -1,    57,    58,    59,    94,    95,    -1,    -1,    -1,
      99,    -1,     4,     5,     6,     7,     8,    -1,   107,    -1,
     109,   110,   111,    -1,    79,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    94,
      95,    -1,    -1,    -1,    99,    37,     4,     5,     6,     7,
       8,    -1,   107,    -1,   109,   110,   111,    -1,    -1,    51,
      -1,    53,    54,    -1,    -1,    57,    -1,    59,    26,    27,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    -1,
      -1,    -1,    -1,    51,    -1,    53,    54,    -1,    -1,    57,
      -1,    59,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   107,    -1,   109,   110,   111,
      -1,    79,    80,    -1,    -1,    47,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    57,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,
      72,   109,   110,   111,    76,    77,    78,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    47,    -1,   109,   110,    -1,
      -1,    -1,    -1,    -1,    -1,    57,    58,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    47,    -1,   109,   110,    -1,
      -1,    -1,    -1,    -1,    -1,    57,    58,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    56,    57,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    -1,    -1,   109,   110,    -1,
      -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      57,    58,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    -1,    -1,   109,   110,    -1,    -1,    76,
      77,    78,    -1,    -1,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    57,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      -1,    -1,   109,   110,    74,    -1,    76,    77,    78,    -1,
      -1,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    57,    58,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,    -1,    -1,   109,
     110,    -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    -1,    -1,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    57,    -1,   109,   110,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    72,
      -1,    -1,    -1,    76,    77,    78,    -1,    -1,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    57,    58,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,    -1,    -1,   109,   110,    -1,    -1,
      76,    77,    78,    -1,    -1,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    57,    58,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,    -1,    -1,   109,   110,    -1,    -1,    76,    77,    78,
      -1,    -1,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    57,    -1,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,    -1,    -1,
     109,   110,    -1,    -1,    76,    77,    78,    -1,    -1,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    57,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    -1,    -1,   109,   110,    -1,
      -1,    -1,    77,    78,    -1,    -1,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    57,
      -1,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,    -1,    -1,   109,   110,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,   109,   110
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   113,   114,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    45,    46,    48,
      50,    51,    52,    53,    54,    55,    57,    59,    79,    80,
      84,    94,    95,    99,   105,   107,   109,   110,   111,   115,
     116,   117,   118,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   137,
     138,   139,   141,   142,   150,   151,   152,   154,   155,   156,
     161,   162,   163,   170,   172,   185,   187,   196,   197,   199,
     206,   207,   208,   210,   212,   220,   221,   222,   223,   225,
     227,   230,   231,   232,   235,   253,   258,   262,   263,   264,
     265,   266,   267,   268,   269,   274,   277,   278,   279,     3,
       3,     1,   119,   264,     1,   266,   267,     1,     3,     1,
       3,    14,     1,   267,     1,   264,   266,   282,     1,   267,
       1,     1,   267,     1,   267,   280,     1,     3,    47,     1,
     267,     1,     6,     1,     6,     1,     3,   267,   259,   275,
       1,     6,     7,     1,   267,   269,     1,     6,     1,     3,
      47,     1,   267,     1,     3,     6,   224,     1,     6,   224,
     226,     1,     6,   228,   229,     1,     6,     1,   267,    58,
     267,   281,    47,   267,    47,    58,    72,   267,   280,   284,
     267,   266,     1,     3,   280,   267,   267,   267,     1,     3,
     280,   267,   267,   267,   267,   136,    32,    34,    48,   118,
     140,   118,   153,   118,   171,   186,   198,    49,   215,   218,
     219,   118,     1,    57,     1,     6,   233,   234,   233,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    75,    88,   268,     3,    57,    71,    76,    77,    78,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   109,   110,    59,    60,   268,     3,
       3,    75,    88,     3,    47,     3,    47,     3,     3,     3,
       3,    47,     3,    47,     3,    47,    88,    75,    91,     3,
       3,     3,     3,     3,     3,     1,    74,    75,     3,   118,
       3,     3,     3,   236,     3,   254,     3,     3,     6,   260,
     261,     1,     6,   213,   214,   276,     3,     3,     3,     3,
       3,     3,    88,     3,    47,     3,     3,    91,     3,     3,
      75,     3,    75,     3,     3,    87,     3,    75,     3,     3,
       3,     1,    58,   267,    56,    58,   267,    58,    47,    72,
       1,    58,     1,    58,    75,    87,    88,     3,     3,     3,
       3,   149,   149,   149,   173,   188,   149,     1,     3,    47,
     149,   216,   217,     3,     1,   213,     3,     3,    75,     9,
     233,     3,   103,   267,     6,   267,   267,   267,   267,   267,
     267,   267,   267,   267,   267,   266,   283,    48,   111,   267,
     271,   280,    56,   280,   267,     1,   267,   267,   267,   267,
     267,   267,   267,   267,   267,   267,   267,     6,   267,   267,
     267,   267,   267,   267,   267,   267,   267,   267,   267,   267,
     267,   267,   267,   103,   267,     6,   264,   267,   267,   264,
       1,   267,     3,   267,   267,     1,    57,   237,   238,     1,
      33,   240,   255,     3,    75,     3,    75,    72,     1,   263,
       1,   267,     6,     6,     4,     6,   229,    58,    58,   267,
     267,    58,   267,   267,   267,     9,   118,    16,    17,   143,
     145,   146,   148,     9,     1,     3,    23,    24,   174,   179,
     181,     1,     3,    23,   179,   189,    30,   200,   201,   202,
     203,     3,    47,     9,   149,   118,   211,     1,    56,     6,
       3,     3,   267,    58,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,    75,    88,   272,   111,     3,     3,
       3,     1,    56,     3,     1,    47,   267,    58,    88,    74,
       3,     3,    47,     3,     3,   213,   246,   240,     3,     6,
     241,   242,     3,   256,     1,   261,   214,   267,     3,     3,
       3,     3,    58,    72,     3,     1,     3,     1,   267,     9,
     144,   147,     3,     3,     1,     4,     6,     7,     8,   183,
     184,     1,     9,   180,     3,     1,     4,     6,   194,   195,
       9,     1,     3,     4,     6,    91,   204,   205,     9,   202,
     149,     3,     9,    56,   209,     3,    47,    58,   266,   267,
     280,     1,    57,   273,     3,   270,     1,   267,    58,   267,
     267,   157,   158,     1,    56,     3,     6,    38,    48,    49,
      93,   207,   247,   248,   250,   251,     3,    57,   243,    75,
       3,   207,   248,   250,   251,   257,   267,     3,     3,     3,
       3,   149,   149,     3,    47,    74,     3,    47,    75,     3,
       3,    47,   182,     3,    47,     3,    47,    75,     3,     3,
     264,     3,    75,    91,     3,     3,    47,    56,     3,     3,
       3,   213,   215,    56,     3,    47,    73,    19,    20,    21,
     118,   159,   160,   164,   166,   168,   118,   239,    88,     3,
       6,     1,     6,    79,   252,     9,     6,    27,   244,   245,
     263,   242,     9,   143,   177,   178,     4,   175,   176,   184,
     149,   118,   192,   193,   190,   191,   195,     3,   205,   264,
       3,     1,    56,   149,   267,     1,     3,    47,     1,     3,
      47,     1,     3,    47,     9,   159,    56,   267,   249,    88,
       3,     6,     3,    75,     3,    56,    75,     3,   149,   118,
     149,   118,   149,   118,   149,   118,     3,     3,     9,     3,
      47,     3,   165,   118,     3,   167,   118,     3,   169,   118,
       3,     3,   215,   267,     6,    79,   245,   149,   149,   149,
     149,     3,     6,     9,     9,     9,     9,     3,     3,     3,
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
#line 208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); ;}
    break;

  case 7:
#line 209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); ;}
    break;

  case 8:
#line 215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      ;}
    break;

  case 10:
#line 221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 11:
#line 226 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 12:
#line 231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 13:
#line 236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 18:
#line 248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 19:
#line 254 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 20:
#line 260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      ;}
    break;

  case 21:
#line 267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); ;}
    break;

  case 22:
#line 268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; ;}
    break;

  case 23:
#line 271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; ;}
    break;

  case 24:
#line 272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; ;}
    break;

  case 25:
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; ;}
    break;

  case 26:
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;;}
    break;

  case 27:
#line 279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   ;}
    break;

  case 28:
#line 284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAssignment( CURRENT_LINE, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 29:
#line 291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); ;}
    break;

  case 51:
#line 317 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; ;}
    break;

  case 52:
#line 319 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); ;}
    break;

  case 53:
#line 324 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 54:
#line 328 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtUnref( LINE, (yyvsp[(1) - (5)].fal_val) );
   ;}
    break;

  case 55:
#line 332 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) );
      ;}
    break;

  case 56:
#line 336 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 57:
#line 340 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), (yyvsp[(5) - (6)].fal_val) );
      ;}
    break;

  case 58:
#line 345 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (6)].fal_val) );
         (yyvsp[(3) - (6)].fal_adecl)->pushFront( (yyvsp[(1) - (6)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtAssignment( LINE, new Falcon::Value((yyvsp[(3) - (6)].fal_adecl)), new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) ) );
      ;}
    break;

  case 70:
#line 369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoAdd( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 71:
#line 376 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSub( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 72:
#line 383 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMul( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 73:
#line 390 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoDiv( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 74:
#line 397 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoMod( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 75:
#line 404 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoPow( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 76:
#line 411 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBAND( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 77:
#line 418 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 78:
#line 425 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoBXOR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 79:
#line 431 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHL( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 80:
#line 437 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (4)].fal_val) );
      (yyval.fal_stat) = new Falcon::StmtAutoSHR( LINE, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
   ;}
    break;

  case 81:
#line 445 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      ;}
    break;

  case 82:
#line 452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 83:
#line 459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      ;}
    break;

  case 84:
#line 467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 85:
#line 468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 86:
#line 469 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; ;}
    break;

  case 87:
#line 473 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 88:
#line 474 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 89:
#line 475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 90:
#line 479 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      ;}
    break;

  case 91:
#line 487 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 92:
#line 494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 93:
#line 504 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 94:
#line 505 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; ;}
    break;

  case 95:
#line 509 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 96:
#line 510 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 99:
#line 517 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      ;}
    break;

  case 102:
#line 527 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); ;}
    break;

  case 103:
#line 532 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      ;}
    break;

  case 105:
#line 544 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 106:
#line 545 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; ;}
    break;

  case 108:
#line 550 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   ;}
    break;

  case 109:
#line 557 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      ;}
    break;

  case 110:
#line 566 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 111:
#line 574 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      ;}
    break;

  case 112:
#line 584 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      ;}
    break;

  case 113:
#line 593 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 114:
#line 600 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>( (yyvsp[(1) - (1)].fal_stat) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 115:
#line 607 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFor *f = static_cast<Falcon::StmtFor *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 116:
#line 615 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 117:
#line 630 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 118:
#line 634 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 119:
#line 639 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for );
         (yyval.fal_stat) = new Falcon::StmtFor( LINE, 0, 0, 0 );
      ;}
    break;

  case 120:
#line 646 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (7)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) );
      ;}
    break;

  case 121:
#line 650 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (9)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, (yyvsp[(2) - (9)].fal_val), (yyvsp[(4) - (9)].fal_val), (yyvsp[(6) - (9)].fal_val), (yyvsp[(8) - (9)].fal_val) );
      ;}
    break;

  case 122:
#line 655 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_for, "", CURRENT_LINE );
         (yyval.fal_stat) = new Falcon::StmtFor( CURRENT_LINE, 0, 0, 0 );
      ;}
    break;

  case 123:
#line 664 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 124:
#line 680 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 125:
#line 688 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 126:
#line 704 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 127:
#line 714 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_forin ); ;}
    break;

  case 130:
#line 723 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 134:
#line 737 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 135:
#line 750 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 136:
#line 758 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 137:
#line 762 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 138:
#line 768 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 139:
#line 774 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
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
#line 786 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 142:
#line 795 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   ;}
    break;

  case 143:
#line 804 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 144:
#line 816 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 145:
#line 818 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 146:
#line 827 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 147:
#line 831 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 148:
#line 843 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 149:
#line 844 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 150:
#line 853 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 151:
#line 857 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 152:
#line 871 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 153:
#line 873 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 154:
#line 882 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); ;}
    break;

  case 155:
#line 886 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 156:
#line 894 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 157:
#line 903 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 158:
#line 905 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 161:
#line 914 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 163:
#line 920 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 165:
#line 930 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 166:
#line 938 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 167:
#line 942 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 169:
#line 954 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 170:
#line 964 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 172:
#line 973 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 176:
#line 987 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 178:
#line 991 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 181:
#line 1003 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 182:
#line 1012 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 183:
#line 1024 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 184:
#line 1035 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 185:
#line 1046 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 186:
#line 1066 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 187:
#line 1074 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 188:
#line 1083 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 189:
#line 1085 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 192:
#line 1094 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 194:
#line 1100 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 196:
#line 1110 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 197:
#line 1119 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 198:
#line 1123 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 200:
#line 1135 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 201:
#line 1145 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 205:
#line 1159 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 206:
#line 1171 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 207:
#line 1192 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_val), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 208:
#line 1196 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 209:
#line 1200 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 210:
#line 1208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 211:
#line 1215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 212:
#line 1225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 214:
#line 1234 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 220:
#line 1254 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 221:
#line 1272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 222:
#line 1292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 223:
#line 1302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 224:
#line 1313 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 227:
#line 1326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 228:
#line 1338 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 229:
#line 1360 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 230:
#line 1361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 231:
#line 1373 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 232:
#line 1379 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 234:
#line 1388 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 235:
#line 1389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 236:
#line 1392 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 238:
#line 1397 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 239:
#line 1398 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 240:
#line 1405 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 244:
#line 1466 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 246:
#line 1483 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 247:
#line 1489 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 248:
#line 1494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 249:
#line 1500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 251:
#line 1509 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 253:
#line 1514 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 254:
#line 1524 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 255:
#line 1527 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 256:
#line 1536 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 257:
#line 1543 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );

         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
      ;}
    break;

  case 258:
#line 1553 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 259:
#line 1559 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 260:
#line 1571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 261:
#line 1581 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 262:
#line 1586 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 263:
#line 1598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 264:
#line 1607 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 265:
#line 1614 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 266:
#line 1622 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 267:
#line 1627 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 268:
#line 1635 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 269:
#line 1639 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 270:
#line 1647 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->imported(true);
      ;}
    break;

  case 271:
#line 1652 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->imported(true);
      ;}
    break;

  case 272:
#line 1664 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 273:
#line 1669 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     ;}
    break;

  case 276:
#line 1682 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      ;}
    break;

  case 277:
#line 1686 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      ;}
    break;

  case 278:
#line 1700 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 279:
#line 1707 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 281:
#line 1715 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 283:
#line 1719 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 285:
#line 1725 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 286:
#line 1729 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 289:
#line 1738 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 290:
#line 1749 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 291:
#line 1783 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 293:
#line 1811 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 296:
#line 1819 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 297:
#line 1820 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 302:
#line 1837 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 303:
#line 1870 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 304:
#line 1875 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
   ;}
    break;

  case 305:
#line 1881 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 306:
#line 1882 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 308:
#line 1888 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      ;}
    break;

  case 309:
#line 1896 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 313:
#line 1906 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 314:
#line 1909 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 316:
#line 1931 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 317:
#line 1955 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 318:
#line 1966 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 319:
#line 1988 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 322:
#line 2018 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 323:
#line 2025 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 324:
#line 2033 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 325:
#line 2039 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 326:
#line 2045 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 327:
#line 2058 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 328:
#line 2098 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 330:
#line 2123 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 334:
#line 2135 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 335:
#line 2138 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 337:
#line 2166 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 338:
#line 2171 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 341:
#line 2186 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 342:
#line 2193 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 343:
#line 2208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 344:
#line 2209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 345:
#line 2210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 346:
#line 2220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 347:
#line 2221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); ;}
    break;

  case 348:
#line 2222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); ;}
    break;

  case 349:
#line 2223 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 350:
#line 2224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 351:
#line 2225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 352:
#line 2230 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 354:
#line 2248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 355:
#line 2249 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 357:
#line 2261 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 358:
#line 2266 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 359:
#line 2271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 360:
#line 2277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 363:
#line 2292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 364:
#line 2293 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 365:
#line 2294 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 366:
#line 2295 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 367:
#line 2296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 368:
#line 2297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 369:
#line 2298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 370:
#line 2299 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 371:
#line 2300 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 372:
#line 2301 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 373:
#line 2302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 374:
#line 2303 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 375:
#line 2304 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 376:
#line 2305 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 377:
#line 2307 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) ); (yyval.fal_val) =
        new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_let, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); ;}
    break;

  case 378:
#line 2309 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 379:
#line 2310 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 380:
#line 2311 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 381:
#line 2312 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 382:
#line 2313 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 383:
#line 2314 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 384:
#line 2315 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 385:
#line 2316 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 386:
#line 2317 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 387:
#line 2318 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 388:
#line 2319 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 389:
#line 2320 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 390:
#line 2321 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 391:
#line 2322 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 392:
#line 2323 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 393:
#line 2324 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 394:
#line 2325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 395:
#line 2326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 396:
#line 2327 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 397:
#line 2328 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 398:
#line 2329 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 399:
#line 2330 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 400:
#line 2331 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 403:
#line 2334 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 404:
#line 2342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 405:
#line 2346 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 406:
#line 2350 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 411:
#line 2358 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 412:
#line 2363 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 413:
#line 2366 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 414:
#line 2369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 415:
#line 2372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 416:
#line 2379 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      ;}
    break;

  case 417:
#line 2385 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      ;}
    break;

  case 418:
#line 2389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 419:
#line 2390 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 420:
#line 2399 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 2432 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->closeFunction();
         ;}
    break;

  case 423:
#line 2443 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      ;}
    break;

  case 424:
#line 2447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 425:
#line 2455 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 426:
#line 2486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            COMPILER->checkLocalUndefined();
            COMPILER->closeFunction();
         ;}
    break;

  case 428:
#line 2500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      ;}
    break;

  case 429:
#line 2509 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 430:
#line 2514 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 431:
#line 2521 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 432:
#line 2528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 433:
#line 2537 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 434:
#line 2539 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 435:
#line 2543 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 436:
#line 2549 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 437:
#line 2551 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 438:
#line 2555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 439:
#line 2563 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 440:
#line 2564 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); ;}
    break;

  case 441:
#line 2566 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      ;}
    break;

  case 442:
#line 2573 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 443:
#line 2574 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 444:
#line 2578 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 445:
#line 2579 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (2)].fal_adecl)->pushBack( (yyvsp[(2) - (2)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (2)].fal_adecl); ;}
    break;

  case 446:
#line 2583 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 447:
#line 2589 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 448:
#line 2596 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 449:
#line 2602 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 450:
#line 2609 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 451:
#line 2610 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6424 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2614 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


